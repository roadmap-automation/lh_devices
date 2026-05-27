from typing import List, Tuple, Literal, Dict, Any, Optional
import logging
import numpy as np
from pydantic import Field

from .lhmethods import MixWithRinse, TransferWithRinse, LHMethodCluster, MethodContainer, MethodsType

from lh_devices.core.bedlayout import Composition, LHBedLayout, Well, WellLocation
from .reservation import reservation_store
from lh_devices.core.formulation import (
    ZERO_VOLUME_TOLERANCE,
    make_target_vector,
    select_wells,
    make_source_matrix,
    solve_formulation as _solve_formulation_core,
)


def get_all_wells_in_zones(layout: LHBedLayout, include_zones: List[str]) -> List[Well]:
    """Returns all wells in the layout belonging to the specified zones (matched by rack_id)."""
    return [w for rack in layout.racks.values()
            for w in rack.wells
            if w.rack_id in include_zones]


def solve_formulation(
    layout: LHBedLayout,
    target_composition: Composition,
    target_volume: float,
    exact_match: bool = True,
    include_zones: List[str] = ['Solvent', 'Stock', 'Samples'],
) -> Dict[str, Any]:
    """Zone-aware wrapper around core solve_formulation."""
    wells = get_all_wells_in_zones(layout, include_zones)
    return _solve_formulation_core(wells, layout, target_composition, target_volume, exact_match)


class Formulation(MethodContainer):

    method_name: Literal['Formulation'] = 'Formulation'
    display_name: Literal['Formulation'] = 'Formulation'
    target_composition: Composition = Field(default_factory=Composition)
    target_volume: float = 0.0
    Target: WellLocation = Field(default_factory=WellLocation)
    include_zones: List[str] = Field(default=['Solvent', 'Stock', 'Samples'])
    exact_match: bool = True
    Extra_Volume: float = 0.1
    Aspirate_Flow_Rate: float = 2.0
    Flow_Rate: float = 2.5
    Use_Liquid_Level_Detection: bool = True

    _formulation_results: Tuple[List[float], List[Well], bool] | None = None

    def _inflated_target(self, layout: LHBedLayout) -> float:
        """Volume to prepare when mixing is needed: accounts for mix overhead, inject overhead,
        and the target well's rack minimum volume."""
        rack_id = self.Target.rack_id or "Mix"
        rack = layout.racks.get(rack_id)
        rack_min = rack.min_volume if rack else 0.0
        return self.target_volume + rack_min + 2 * self.Extra_Volume

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        # Pass 1: check if the composition already exists at injection-overhead-adjusted volume.
        # The core solver adds rack_min_source internally when verifying each source well.
        result = solve_formulation(
            layout=layout,
            target_composition=self.target_composition,
            target_volume=self.target_volume + self.Extra_Volume,
            exact_match=self.exact_match,
            include_zones=self.include_zones,
        )

        if result['success'] and len(result['wells']) == 1:
            # Case (a): composition exists in one well; inject overhead already accounted for.
            self._formulation_results = result['volumes'], result['wells'], True
            logging.info(self._formulation_results)
            return self._formulation_results

        if not result['success']:
            logging.error(result['error'])
            self._formulation_results = [], [], False
            return self._formulation_results

        # Case (b): mixing needed — re-solve with fully inflated target volume.
        # Inflated = target_volume + rack_min_mix + Extra_Volume_mix + Extra_Volume_inject
        result2 = solve_formulation(
            layout=layout,
            target_composition=self.target_composition,
            target_volume=self._inflated_target(layout),
            exact_match=self.exact_match,
            include_zones=self.include_zones,
        )
        if not result2['success']:
            logging.error(result2['error'])
        self._formulation_results = result2['volumes'], result2['wells'], result2['success']
        logging.info(self._formulation_results)
        return self._formulation_results

    def get_formulation_results(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        if self._formulation_results is None:
            self._formulation_results = self.formulate(layout)
        return self._formulation_results

    def get_expected_composition(self, layout: LHBedLayout) -> Composition:
        volumes, wells, success = self.get_formulation_results(layout)
        if success:
            mix_well = Well(rack_id='', volume=0, composition=Composition(), well_number=0)
            for volume, well in zip(volumes, wells):
                mix_well.mix_with(volume, well.composition)
            return mix_well.composition
        return Composition()

    def get_target_well(self, layout: LHBedLayout) -> WellLocation:
        volumes, wells, success = self.get_formulation_results(layout)
        if success:
            if len(volumes) > 1:
                return self.Target
            else:
                return WellLocation(rack_id=wells[0].rack_id,
                                    well_number=wells[0].well_number,
                                    expected_composition=self.target_composition)

    def get_methods(self, layout: LHBedLayout, sample_id: str | None = None) -> List[MethodsType]:
        methods = []
        volumes, wells, success = self.get_formulation_results(layout)

        if self.Target.id is not None and sample_id is not None:
            if success and len(volumes) == 1:
                # Composition found in a single well — register it and skip transfers.
                existing = WellLocation(
                    rack_id=wells[0].rack_id,
                    well_number=wells[0].well_number,
                    expected_composition=self.target_composition,
                )
                reservation_store.reserve_reference(sample_id, self.Target.id, existing)
                return []
            elif success:
                # Multi-source: claim an empty Mix well for the formulation target.
                claimed = reservation_store.reserve_claim(
                    sample_id, self.Target.id, layout, rack_id="Mix"
                )
                if claimed is None:
                    raise RuntimeError(
                        f"No empty Mix well available for allocation {self.Target.id!r}"
                    )
                self.Target = claimed

        if success:
            sort_index = np.argsort(volumes)[::-1]
            sorted_volumes = [volumes[si] for si in sort_index]
            sorted_wells: list[Well] = [wells[si] for si in sort_index]

            for volume, well in zip(sorted_volumes, sorted_wells):
                methods.append(TransferWithRinse(
                    Source=WellLocation(rack_id=well.rack_id, well_number=well.well_number),
                    Target=self.Target,
                    Volume=volume,
                    Aspirate_Flow_Rate=self.Aspirate_Flow_Rate,
                    Flow_Rate=self.Flow_Rate,
                    Use_Liquid_Level_Detection=self.Use_Liquid_Level_Detection,
                ))

            if len(volumes) > 1:
                total_volume = sum(volumes)
                mix_volume = max(0.9 * total_volume, 0.1)
                methods.append(MixWithRinse(
                    Target=self.Target,
                    Volume=mix_volume,
                    Aspirate_Flow_Rate=self.Aspirate_Flow_Rate,
                    Flow_Rate=self.Flow_Rate,
                    Use_Liquid_Level_Detection=self.Use_Liquid_Level_Detection,
                ))

        return [] if not methods else [LHMethodCluster(methods=methods)]

    def make_source_matrix(self, source_names, wells, source_units):
        return make_source_matrix(source_names, wells, source_units)

    def make_target_vector(self):
        return make_target_vector(self.target_composition)

    def select_wells(self, wells, target_names):
        return select_wells(wells, target_names, self.exact_match)

    def get_all_wells(self, layout):
        return get_all_wells_in_zones(layout, self.include_zones)


class SoluteFormulation(Formulation):
    """Subclass of Formulation. Solutes are specified; remaining volume is filled with diluent."""

    method_name: Literal['SoluteFormulation'] = 'SoluteFormulation'
    display_name: Literal['SoluteFormulation'] = 'SoluteFormulation'
    exact_match: bool = False
    diluent: Composition = Field(default_factory=Composition)

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        # SoluteFormulation always involves ≥2 transfers (solutes + diluent), so always case (b).
        # Solve directly at the inflated volume to keep the diluent top-up consistent.
        inflated = self._inflated_target(layout)

        result = _solve_formulation_core(
            wells=get_all_wells_in_zones(layout, self.include_zones),
            layout=layout,
            target_composition=self.target_composition,
            target_volume=inflated,
            exact_match=False,
        )

        if not result['success']:
            logging.error(result['error'])
            self._formulation_results = [], [], False
            return self._formulation_results

        volumes, wells = result['volumes'], result['wells']

        diluent_well = next(
            (w for w in self.get_all_wells(layout) if w.composition == self.diluent),
            None,
        )
        if diluent_well is None:
            logging.error('Diluent (%s) not available on bed', self.diluent)
            self._formulation_results = [], [], False
            return self._formulation_results

        diluent_volume = inflated - sum(volumes)
        if not np.isclose(diluent_volume, 0.0, atol=ZERO_VOLUME_TOLERANCE):
            if diluent_volume < 0:
                logging.error('Diluent volume less than zero; should never happen')
                self._formulation_results = [], [], False
                return self._formulation_results
            volumes += [diluent_volume]
            wells += [diluent_well]

        self._formulation_results = volumes, wells, True
        logging.info(self._formulation_results)
        return self._formulation_results
