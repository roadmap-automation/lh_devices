from typing import List, Tuple, Literal, Dict, Any, Optional
from copy import copy
import logging
import numpy as np
from pydantic import Field, validator, SerializeAsAny

from .lhmethods import MixMethod, MixWithRinse, TransferMethod, TransferWithRinse, LHMethodCluster, MethodContainer, MethodsType
from .layoutmap import Zone, LayoutWell2ZoneWell

from lh_devices.core.bedlayout import Composition, LHBedLayout, Well, WellLocation
from lh_devices.core.formulation import (
    ZERO_VOLUME_TOLERANCE,
    make_target_vector,
    select_wells,
    make_source_matrix,
    solve_formulation as _solve_formulation_core,
)

# Local template class registry for deserialization in validate_templates
_TEMPLATE_CLASSES: dict[str, type] = {
    TransferWithRinse.model_fields['method_name'].default: TransferWithRinse,
    MixWithRinse.model_fields['method_name'].default: MixWithRinse,
}


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
    include_zones: List[str] = Field(default_factory=lambda: ['Solvent', 'Stock', 'Samples'])
    exact_match: bool = True
    transfer_template: SerializeAsAny[TransferMethod] = Field(default_factory=TransferWithRinse)
    mix_template: SerializeAsAny[MixMethod] = Field(default_factory=MixWithRinse)

    _formulation_results: Tuple[List[float], List[Well], bool] | None = None

    @validator('mix_template', 'transfer_template', pre=True)
    def validate_templates(cls, v):
        if isinstance(v, dict):
            method_cls = _TEMPLATE_CLASSES.get(v.get('method_name', ''), TransferWithRinse)
            return method_cls(**v)
        return v

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        result = solve_formulation(
            layout=layout,
            target_composition=self.target_composition,
            target_volume=self.target_volume,
            exact_match=self.exact_match,
            include_zones=self.include_zones,
        )
        if not result['success']:
            logging.error(result['error'])
        self._formulation_results = result['volumes'], result['wells'], result['success']
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

    def get_methods(self, layout: LHBedLayout) -> List[MethodsType]:
        methods = []
        volumes, wells, success = self.get_formulation_results(layout)
        if success:
            sort_index = np.argsort(volumes)[::-1]
            sorted_volumes = [volumes[si] for si in sort_index]
            sorted_wells: list[Well] = [wells[si] for si in sort_index]

            for volume, well in zip(sorted_volumes, sorted_wells):
                new_transfer = copy(self.transfer_template)
                new_transfer.Source = WellLocation(rack_id=well.rack_id, well_number=well.well_number)
                new_transfer.Target = self.Target
                new_transfer.Volume = volume
                methods.append(new_transfer)

            if len(volumes) > 1:
                total_volume = sum(volumes)
                min_mix_volume = 0.1
                mix_volume = max(0.9 * total_volume, min_mix_volume)
                new_mix = copy(self.mix_template)
                new_mix.Target = self.Target
                new_mix.Volume = mix_volume
                methods.append(new_mix)

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
        volumes, wells, success = super().formulate(layout)

        if not success:
            return [], [], False

        diluent_well = next((well for well in self.get_all_wells(layout) if well.composition == self.diluent), None)

        if diluent_well is None:
            logging.error('Diluent (%s) not available on bed', self.diluent)
            return [], [], False

        diluent_volume = self.target_volume - sum(volumes)

        if not np.isclose(diluent_volume, 0.0, atol=ZERO_VOLUME_TOLERANCE):
            volumes += [diluent_volume]
            wells += [diluent_well]

            if diluent_volume < 0:
                logging.error('Diluent volume less than zero; should never happen')
                return [], [], False

        return volumes, wells, True
