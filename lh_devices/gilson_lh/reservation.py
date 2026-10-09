"""Well reservation map: claim (blocking, empty mix well) vs reference (non-blocking, stock well).

Persisted to SQLite so reservations survive a process restart.
Cleanup is triggered by release_sample() on SUBPROTOCOL_COMPLETED.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

from lh_devices.core.bedlayout import LHBedLayout, WellLocation

from .app_config import config

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS reservations (
    sample_id TEXT NOT NULL,
    uuid      TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK(kind IN ('claim', 'reference')),
    rack_id   TEXT,
    well_number INTEGER,
    PRIMARY KEY (sample_id, uuid)
) WITHOUT ROWID;
"""


class WellReservationStore:
    """Thread-safe (single-process) SQLite-backed reservation map.

    Two kinds:
      claim     — blocking. Assigns the next empty well in a rack to (sample_id, uuid).
                  The well is stamped (well.id = uuid) so WellResolver can find it via layout scan.
      reference — non-blocking. Records an existing well as associated with (sample_id, uuid)
                  so WellResolver can look it up via SQLite without touching the layout.
    """

    def __init__(self, db_path: Path = config.reservation_path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(self.db_path.parent, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reserve_claim(self, sample_id: str, uuid: str, layout: LHBedLayout, required_volume: float = 0.0) -> WellLocation | None:
        """Assign the next empty well in an allow_mixing rack to (sample_id, uuid).

        Selects the best-fit rack: smallest max_volume >= required_volume among
        racks with allow_mixing=True that have at least one unclaimed empty well.
        Falls back to the smallest allow_mixing rack if none fits the volume (overflow
        is caught at execute time by MethodError checks).

        If (sample_id, uuid) is already claimed, return the existing location.
        Returns None if no allow_mixing rack with an empty well is available.
        """
        existing = self._lookup(sample_id, uuid)
        if existing is not None:
            return existing

        candidates = [
            (name, rack)
            for name, rack in layout.racks.items()
            if rack.allow_mixing and layout.find_next_empty(name) is not None
        ]

        if not candidates:
            logger.error("reserve_claim: no allow_mixing rack with empty wells available")
            return None

        fitting = [(name, rack) for name, rack in candidates if rack.max_volume >= required_volume]
        if fitting:
            chosen_name, _ = min(fitting, key=lambda x: x[1].max_volume)
        else:
            chosen_name, _ = min(candidates, key=lambda x: x[1].max_volume)
            logger.warning(
                "reserve_claim: no allow_mixing rack fits required volume %.3f mL; "
                "using %s (max_volume=%.3f) — overflow check will run at execute time",
                required_volume, chosen_name, layout.racks[chosen_name].max_volume,
            )

        next_empty = layout.find_next_empty(chosen_name)

        # Mark the well in the layout with this uuid so infer_location can find it
        well, _ = layout.get_well_and_rack(next_empty.rack_id, next_empty.well_number)
        well.id = uuid

        loc = WellLocation(rack_id=next_empty.rack_id, well_number=next_empty.well_number, id=uuid)
        self._persist(sample_id, uuid, 'claim', loc)
        logger.info("reserve_claim: %s/%s → %s/%s", sample_id, uuid, loc.rack_id, loc.well_number)
        return loc

    def reserve_reference(self, sample_id: str, uuid: str, well_location: WellLocation) -> WellLocation:
        """Record an existing well as a reference for (sample_id, uuid).

        If already recorded, return the existing location.
        Does not stamp the layout well — multiple samples may reference the same
        stock well with different uuids, so resolution is always done via SQLite.
        """
        existing = self._lookup(sample_id, uuid)
        if existing is not None:
            return existing

        self._persist(sample_id, uuid, 'reference', well_location)
        logger.info("reserve_reference: %s/%s → %s/%s", sample_id, uuid, well_location.rack_id, well_location.well_number)
        return well_location

    def lookup(self, sample_id: str, uuid: str) -> WellLocation | None:
        """Look up a reservation by (sample_id, uuid)."""
        return self._lookup(sample_id, uuid)

    def is_claimed(self, rack_id: str, well_number: int) -> bool:
        """Return True if this well is actively claimed by any sample.

        Used to guard against reusing a Mix well that belongs to an in-flight
        subprotocol. An orphaned well (released from DB but well.id still set
        in the layout) returns False and is safe to reuse.
        """
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT 1 FROM reservations WHERE kind='claim' AND rack_id=? AND well_number=?",
                (rack_id, well_number),
            ).fetchone()
        return row is not None

    def release_sample(self, sample_id: str) -> None:
        """Remove all reservations for sample_id. Called on SUBPROTOCOL_COMPLETED."""
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM reservations WHERE sample_id = ?", (sample_id,))
        logger.info("reservation: released all entries for sample %s", sample_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lookup(self, sample_id: str, uuid: str) -> WellLocation | None:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT rack_id, well_number FROM reservations WHERE sample_id=? AND uuid=?",
                (sample_id, uuid),
            ).fetchone()
        if row is None:
            return None
        return WellLocation(rack_id=row[0], well_number=row[1], id=uuid)

    def _persist(self, sample_id: str, uuid: str, kind: str, loc: WellLocation) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT OR REPLACE INTO reservations(sample_id, uuid, kind, rack_id, well_number) VALUES (?,?,?,?,?)",
                (sample_id, uuid, kind, loc.rack_id, loc.well_number),
            )


reservation_store = WellReservationStore()
