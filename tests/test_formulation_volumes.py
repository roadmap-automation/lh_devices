"""Unit tests for Formulation volume inflation logic.

Verifies that formulate() accounts for:
  - Injection overhead (Extra_Volume) in all cases
  - Mix overhead (Extra_Volume) and target rack min_volume when mixing is needed
  - SoluteFormulation always prepares at the inflated volume
"""
import pytest

from lh_devices.core.bedlayout import (
    Composition, LHBedLayout, Rack, Solvent, Solute, Well, WellLocation,
)
from lh_devices.gilson_lh.formulation import Formulation, SoluteFormulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rack(min_volume: float = 0.0, max_volume: float = 50.0) -> Rack:
    return Rack(
        columns=1, rows=10, max_volume=max_volume, min_volume=min_volume,
        style='grid', wells=[], height=400, width=100,
        x_translate=0, y_translate=0, shape='circle',
    )


def pure(name: str) -> Composition:
    return Composition(solvents=[Solvent(name=name, fraction=1.0)])


def mix50(a: str, b: str) -> Composition:
    return Composition(solvents=[Solvent(name=a, fraction=0.5), Solvent(name=b, fraction=0.5)])


def build_layout(
    source_wells: list[Well],
    source_rack_name: str = 'Stock',
    source_min_vol: float = 0.0,
    mix_min_vol: float = 0.2,
) -> LHBedLayout:
    layout = LHBedLayout()
    source_rack = make_rack(min_volume=source_min_vol)
    source_rack.wells = source_wells
    layout.racks[source_rack_name] = source_rack

    mix_rack = make_rack(min_volume=mix_min_vol)
    mix_rack.wells = [Well(rack_id='Mix', well_number=1, composition=Composition(), volume=0.0)]
    layout.racks['Mix'] = mix_rack

    return layout


# ---------------------------------------------------------------------------
# Case (a): composition exists in a single source well
# ---------------------------------------------------------------------------

class TestCaseA:
    """Composition exists in one well — inject overhead is added, no mixing."""

    def test_volumes_include_inject_overhead(self):
        target_comp = pure('D2O')
        source_well = Well(rack_id='Stock', well_number=1,
                           composition=target_comp, volume=5.0)
        layout = build_layout(
            source_wells=[source_well],
            source_min_vol=0.0,
            mix_min_vol=0.2,
        )
        f = Formulation(
            target_composition=target_comp,
            target_volume=1.0,
            Extra_Volume=0.1,
            include_zones=['Stock'],
        )
        volumes, wells, success = f.formulate(layout)

        assert success
        assert len(wells) == 1, "Single-source composition must not trigger mixing"
        # solver received target_volume + Extra_Volume = 1.1; that's what the source is reserved for
        assert abs(sum(volumes) - 1.1) < 1e-6, f"Expected 1.1, got {sum(volumes)}"

    def test_fails_when_source_too_shallow(self):
        """Source well volume < target_volume + Extra_Volume + rack_min → fail."""
        target_comp = pure('H2O')
        source_well = Well(rack_id='Stock', well_number=1,
                           composition=target_comp, volume=1.0)
        layout = build_layout(
            source_wells=[source_well],
            source_min_vol=0.5,   # requires 1.1 + 0.5 = 1.6 but only 1.0 available
        )
        f = Formulation(
            target_composition=target_comp,
            target_volume=1.0,
            Extra_Volume=0.1,
            include_zones=['Stock'],
        )
        volumes, wells, success = f.formulate(layout)
        assert not success


# ---------------------------------------------------------------------------
# Case (b): composition requires mixing from multiple source wells
# ---------------------------------------------------------------------------

class TestCaseB:
    """Two-source mix — total prepared volume includes rack_min + 2×Extra_Volume."""

    def _make_mix_layout(self, source_vol: float = 10.0, source_min: float = 0.0,
                          mix_min: float = 0.2, extra: float = 0.1) -> tuple:
        target_comp = mix50('D2O', 'H2O')
        well_d2o = Well(rack_id='Stock', well_number=1,
                        composition=pure('D2O'), volume=source_vol)
        well_h2o = Well(rack_id='Stock', well_number=2,
                        composition=pure('H2O'), volume=source_vol)
        layout = build_layout(
            source_wells=[well_d2o, well_h2o],
            source_min_vol=source_min,
            mix_min_vol=mix_min,
        )
        f = Formulation(
            target_composition=target_comp,
            target_volume=1.0,
            Extra_Volume=extra,
            Target=WellLocation(rack_id='Mix'),
            include_zones=['Stock'],
        )
        return f, layout, mix_min, extra

    def test_inflated_volume(self):
        """sum(volumes) == target_volume + rack_min_mix + 2*Extra_Volume."""
        f, layout, mix_min, extra = self._make_mix_layout()
        volumes, wells, success = f.formulate(layout)

        assert success
        assert len(wells) == 2, "Two-source mix expected"
        expected = 1.0 + mix_min + 2 * extra   # 1.0 + 0.2 + 0.2 = 1.4
        assert abs(sum(volumes) - expected) < 1e-6, (
            f"Expected inflated volume {expected}, got {sum(volumes)}"
        )

    def test_fails_when_source_insufficient_after_inflation(self):
        """Source wells only have enough for target_volume but not inflated volume → fail."""
        target_comp = mix50('D2O', 'H2O')
        # Each source must provide 0.7 mL of inflated 1.4 total,
        # but well only has 0.6 mL (with min_vol=0.0, 0.7 > 0.6 → fail)
        well_d2o = Well(rack_id='Stock', well_number=1,
                        composition=pure('D2O'), volume=0.6)
        well_h2o = Well(rack_id='Stock', well_number=2,
                        composition=pure('H2O'), volume=0.6)
        layout = build_layout(source_wells=[well_d2o, well_h2o])
        f = Formulation(
            target_composition=target_comp,
            target_volume=1.0,
            Extra_Volume=0.1,
            Target=WellLocation(rack_id='Mix'),
            include_zones=['Stock'],
        )
        volumes, wells, success = f.formulate(layout)
        assert not success

    def test_custom_extra_volume(self):
        """Extra_Volume=0.2 → inflated = 1.0 + 0.2 + 2*0.2 = 1.6."""
        f, layout, _, _ = self._make_mix_layout(extra=0.2)
        f.Extra_Volume = 0.2
        f._formulation_results = None  # clear any cached result
        volumes, wells, success = f.formulate(layout)
        assert success
        expected = 1.0 + 0.2 + 2 * 0.2  # 1.6
        assert abs(sum(volumes) - expected) < 1e-6


# ---------------------------------------------------------------------------
# SoluteFormulation
# ---------------------------------------------------------------------------

class TestSoluteFormulation:
    """SoluteFormulation always inflates, diluent fills up to inflated total."""

    def test_diluent_fills_inflated_total(self):
        d2o = pure('D2O')
        peptide_comp = Composition(solutes=[Solute(name='peptide', concentration=0.01, units='mM')])
        full_comp = Composition(
            solvents=[Solvent(name='D2O', fraction=1.0)],
            solutes=[Solute(name='peptide', concentration=0.01, units='mM')],
        )

        # Stock well with 10 mM peptide stock (will require a tiny volume)
        stock_comp = Composition(
            solvents=[Solvent(name='D2O', fraction=1.0)],
            solutes=[Solute(name='peptide', concentration=10.0, units='mM')],
        )
        peptide_well = Well(rack_id='Stock', well_number=1,
                            composition=stock_comp, volume=5.0)
        # Diluent well: pure D2O
        diluent_well = Well(rack_id='Solvent', well_number=1,
                            composition=d2o, volume=50.0)

        layout = LHBedLayout()
        layout.racks['Stock'] = make_rack()
        layout.racks['Stock'].wells = [peptide_well]
        layout.racks['Solvent'] = make_rack()
        layout.racks['Solvent'].wells = [diluent_well]
        mix_rack = make_rack(min_volume=0.2)
        mix_rack.wells = []
        layout.racks['Mix'] = mix_rack

        sf = SoluteFormulation(
            target_composition=peptide_comp,
            target_volume=1.0,
            Extra_Volume=0.1,
            diluent=d2o,
            Target=WellLocation(rack_id='Mix'),
            include_zones=['Stock', 'Solvent'],
        )
        volumes, wells, success = sf.formulate(layout)

        assert success
        inflated = 1.0 + 0.2 + 2 * 0.1  # 1.4
        assert abs(sum(volumes) - inflated) < 1e-4, (
            f"Expected total volume {inflated}, got {sum(volumes)}"
        )
        assert len(volumes) == 2, "Expect two wells: solute + diluent"
