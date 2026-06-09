"""WellResolver: resolves allocation handles (WellLocation.id) to concrete rack/well coordinates.

Three-tier lookup:
  1. Layout scan   — claim wells stamped by reserve_claim; unique per (sample, uuid).
  2. Reservation store — reference wells (shared stock/solvent) keyed by (sample_id, uuid).
  3. Claim next empty — legacy fallback for wells not yet reserved.
"""

from lh_devices.core.bedlayout import LHBedLayout, WellLocation


class WellResolver:
    def __init__(self, layout: LHBedLayout, sample_id: str) -> None:
        self._layout = layout
        self._sample_id = sample_id

    def resolve(self, well: WellLocation) -> WellLocation:
        """Populate rack_id/well_number on well from its id. Mutates in place and returns well."""
        if well.id is None:
            return well

        # 1. Layout scan — claim wells are stamped by reserve_claim
        match = next((w for w in self._layout.get_all_wells() if w.id == well.id), None)
        if match is not None:
            well.rack_id, well.well_number = match.rack_id, match.well_number
            return well

        # 2. Reservation store — reference wells shared across samples
        if self._sample_id:
            from .reservation import reservation_store
            ref = reservation_store.lookup(self._sample_id, well.id)
            if ref is not None:
                well.rack_id, well.well_number = ref.rack_id, ref.well_number
                return well

        # 3. Claim next empty (legacy fallback)
        next_empty = self._layout.find_next_empty(well.rack_id)
        if next_empty is not None:
            target_well, _ = self._layout.get_well_and_rack(next_empty.rack_id, next_empty.well_number)
            target_well.id = well.id
            well.rack_id, well.well_number = next_empty.rack_id, next_empty.well_number

        return well
