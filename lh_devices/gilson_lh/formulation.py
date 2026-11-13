from typing import List, Tuple, Literal, Dict, Any, Optional
import logging
import numpy as np
from pydantic import Field

from lh_devices.methods import MethodException
from .lhmethods import MixWithRinse, TransferWithRinse, LHMethodCluster, MethodContainer, MethodType, MethodsType

from lh_devices.core.bedlayout import Composition, LHBedLayout, Well, WellLocation
from .reservation import reservation_store
from lh_devices.core.formulation import (
    ZERO_VOLUME_TOLERANCE,
    effective_extra_volume,
    make_target_vector,
    select_wells,
    make_source_matrix,
    solve_formulation as _solve_formulation_core,
)


def _solutes_match(target: Composition, candidate: Composition) -> bool:
    """Return True if candidate has exactly the same solutes as target (solvents ignored)."""
    target_solutes = {s.name: s for s in target.solutes if s.concentration > 0}
    candidate_solutes = {s.name: s for s in candidate.solutes if s.concentration > 0}
    if set(target_solutes) != set(candidate_solutes):
        return False
    return all(target_solutes[name] == candidate_solutes[name] for name in target_solutes)


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
    method_type: Literal[MethodType.PREPARE] = MethodType.PREPARE
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
    _failure_reason: str | None = None

    def _available_wells(self, layout: LHBedLayout) -> List[Well]:
        """Wells in include_zones, excluding any actively claimed by another sample."""
        return [
            w for w in get_all_wells_in_zones(layout, self.include_zones)
            if not reservation_store.is_claimed(w.rack_id, w.well_number)
        ]

    def _solve(self, layout: LHBedLayout, target_volume: float) -> Dict[str, Any]:
        """Run the core solver against available (non-claimed) wells."""
        return _solve_formulation_core(
            self._available_wells(layout), layout,
            self.target_composition.stripped(), target_volume, self.exact_match,
            configured_extra_volume=self.Extra_Volume,
        )

    def _inflated_target(self, layout: LHBedLayout) -> float:
        """Volume to prepare when mixing is needed: accounts for mix overhead, inject overhead,
        and the target well's rack minimum volume."""
        rack_id = self.Target.rack_id
        if not rack_id:
            # Pre-claim: estimate rack_min from the first allow_mixing rack available
            rack_id = next(
                (name for name, r in layout.racks.items() if r.allow_mixing),
                None,
            )
        rack = layout.racks.get(rack_id) if rack_id else None
        rack_min = rack.min_volume if rack else 0.0
        return self.target_volume + rack_min + 2 * self.Extra_Volume

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        # Pass 1: check if the composition already exists at injection-overhead-adjusted volume.
        result = self._solve(layout, self.target_volume + self.Extra_Volume)

        if result['success'] and len(result['wells']) == 1:
            # Case (a): composition exists in one well; inject overhead already accounted for.
            self._formulation_results = result['volumes'], result['wells'], True
            logging.info(self._formulation_results)
            return self._formulation_results

        if not result['success']:
            logging.error(result['error'])
            self._failure_reason = result.get('failure_reason')
            self._formulation_results = [], [], False
            return self._formulation_results

        # Case (b): mixing needed — re-solve with fully inflated target volume.
        result2 = self._solve(layout, self._inflated_target(layout))
        if not result2['success']:
            logging.error(result2['error'])
            self._failure_reason = result2.get('failure_reason')
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

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

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
                    sample_id, self.Target.id, layout,
                    required_volume=self._inflated_target(layout),
                )
                if claimed is None:
                    raise MethodException(
                        f"No empty Mix well available for allocation {self.Target.id!r}",
                        retry=True,
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
                rack = layout.racks.get(self.Target.rack_id or "")
                rack_min = rack.min_volume if rack else 0.0
                # Subtract Extra_Volume because the LH adds it back during the mix aspirate.
                safe_headroom = total_volume - rack_min - self.Extra_Volume
                if safe_headroom < ZERO_VOLUME_TOLERANCE:
                    logging.warning(
                        "Skipping mix step: safe_headroom=%.4f mL (total=%.3f, "
                        "rack_min=%.3f, extra=%.3f).",
                        safe_headroom, total_volume, rack_min, self.Extra_Volume,
                    )
                else:
                    mix_volume = min(max(0.9 * total_volume, 0.5 * rack_min), safe_headroom)
                    if safe_headroom < 0.5 * rack_min:
                        logging.warning(
                            "Mix volume limited to %.3f mL (safe headroom=%.3f mL, "
                            "rack_min=%.3f mL); mixing efficiency may be reduced.",
                            mix_volume, safe_headroom, rack_min,
                        )
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


def _composition_matches(target: Composition, candidate: Composition, exact_match: bool) -> bool:
    if exact_match:
        return candidate == target
    return _solutes_match(target, candidate)


class DilutionFactorFormulation(Formulation):
    """Dilutes a stock composition by a fixed factor using a diluent.

    source_composition: composition of the stock well to look for on the bed.
    dilution_factor == 1: reserve that well directly (no mixing).
    dilution_factor > 1: transfer target_volume/factor from the stock well, top up with diluent.
    dilution_factor < 1: always fails (cannot concentrate).

    Both stock and diluent volumes are checked against ZERO_VOLUME_TOLERANCE before any
    well search, so mis-specified factors are caught early with a clear error.
    """

    method_name: Literal['DilutionFactorFormulation'] = 'DilutionFactorFormulation'
    display_name: Literal['DilutionFactorFormulation'] = 'DilutionFactorFormulation'
    exact_match: bool = True
    source_composition: Composition = Field(default_factory=Composition)
    diluent: Composition = Field(default_factory=Composition)
    dilution_factor: float = 1.0

    def _formulate_with_factor(self, layout: LHBedLayout, factor: float) -> Tuple[List[float], List[Well], bool]:
        """Core dilution logic shared with DilutionToConcentrationFormulation."""
        cls = type(self).__name__

        if factor < 1.0 - ZERO_VOLUME_TOLERANCE:
            logging.error('%s: dilution_factor %.4f < 1 (cannot concentrate)', cls, factor)
            self._failure_reason = 'invalid_factor'
            self._formulation_results = [], [], False
            return self._formulation_results

        needed = self.target_volume + self.Extra_Volume

        if np.isclose(factor, 1.0, atol=ZERO_VOLUME_TOLERANCE):
            source_well = next(
                (w for w in self._available_wells(layout)
                 if _composition_matches(self.source_composition, w.composition, self.exact_match)
                 and w.volume >= needed + layout.racks[w.rack_id].min_volume),
                None,
            )
            if source_well is None:
                logging.error(
                    '%s: no source well matching source_composition with sufficient volume '
                    '(%.3f mL) for factor=1', cls, needed,
                )
                self._failure_reason = 'insufficient_volume'
                self._formulation_results = [], [], False
                return self._formulation_results
            self._formulation_results = [needed], [source_well], True
            logging.info('%s factor=1: %s', cls, self._formulation_results)
            return self._formulation_results

        # factor > 1: validate minimum pipettable volumes before searching wells
        stock_check = self.target_volume / factor
        diluent_check = self.target_volume - stock_check
        if stock_check < ZERO_VOLUME_TOLERANCE:
            logging.error(
                '%s: stock volume %.4f mL at factor=%.4f is below minimum pipettable '
                'volume %.4f mL (target_volume=%.4f mL) — reduce factor or increase target volume',
                cls, stock_check, factor, ZERO_VOLUME_TOLERANCE, self.target_volume,
            )
            self._failure_reason = 'below_minimum_volume'
            self._formulation_results = [], [], False
            return self._formulation_results
        if diluent_check < ZERO_VOLUME_TOLERANCE:
            logging.error(
                '%s: diluent volume %.4f mL at factor=%.4f is below minimum pipettable '
                'volume %.4f mL — increase target volume or reduce dilution factor',
                cls, diluent_check, factor, ZERO_VOLUME_TOLERANCE,
            )
            self._failure_reason = 'below_minimum_volume'
            self._formulation_results = [], [], False
            return self._formulation_results

        inflated = self._inflated_target(layout)
        stock_volume = inflated / factor
        diluent_volume = inflated - stock_volume

        source_well = next(
            (w for w in self._available_wells(layout)
             if _composition_matches(self.source_composition, w.composition, self.exact_match)
             and w.volume >= stock_volume + layout.racks[w.rack_id].min_volume),
            None,
        )
        if source_well is None:
            logging.error(
                '%s: no source well matching source_composition with sufficient volume (%.3f mL)',
                cls, stock_volume,
            )
            self._failure_reason = 'insufficient_volume'
            self._formulation_results = [], [], False
            return self._formulation_results

        diluent_well = next(
            (w for w in self._available_wells(layout)
             if w.composition == self.diluent
             and w.volume >= diluent_volume + layout.racks[w.rack_id].min_volume),
            None,
        )
        if diluent_well is None:
            logging.error(
                '%s: no diluent well (%s) with sufficient volume (%.3f mL)',
                cls, self.diluent, diluent_volume,
            )
            self._failure_reason = 'insufficient_volume'
            self._formulation_results = [], [], False
            return self._formulation_results

        self._formulation_results = [stock_volume, diluent_volume], [source_well, diluent_well], True
        logging.info('%s factor=%.4f: %s', cls, factor, self._formulation_results)
        return self._formulation_results

    def get_target_well(self, layout: LHBedLayout) -> WellLocation:
        volumes, wells, success = self.get_formulation_results(layout)
        if success:
            if len(volumes) > 1:
                return self.Target
            else:
                return WellLocation(rack_id=wells[0].rack_id,
                                    well_number=wells[0].well_number,
                                    expected_composition=self.source_composition)

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        return self._formulate_with_factor(layout, self.dilution_factor)


class DilutionToConcentrationFormulation(DilutionFactorFormulation):
    """Dilutes a stock composition until one or more solutes reach a target concentration.

    source_composition: composition of the stock well to look for on the bed (e.g. A+B at stock
        concentrations). Used for both well matching and deriving the dilution factor.
    target_composition: specifies the desired final concentration of one or more solutes
        (e.g. just B at B_target). The dilution factor is computed as stock_conc / target_conc
        for each specified solute. If multiple solutes are given they must all imply the same
        factor (consistent dilution); inconsistency is caught early. Units are assumed to match.
    diluent: composition to top up with after taking the stock fraction.
    """

    method_name: Literal['DilutionToConcentrationFormulation'] = 'DilutionToConcentrationFormulation'
    display_name: Literal['DilutionToConcentrationFormulation'] = 'DilutionToConcentrationFormulation'

    def _compute_factor_from_target(self) -> float | None:
        target_solutes = {s.name: s for s in self.target_composition.stripped().solutes}
        stock_solutes = {s.name: s for s in self.source_composition.stripped().solutes}

        if not target_solutes:
            logging.error('DilutionToConcentrationFormulation: target_composition has no solutes')
            return None

        factors = []
        for name, target_solute in target_solutes.items():
            if name not in stock_solutes:
                logging.error(
                    'DilutionToConcentrationFormulation: target solute %r not found in source_composition',
                    name,
                )
                return None
            if target_solute.concentration <= 0:
                logging.error(
                    'DilutionToConcentrationFormulation: target concentration for %r must be > 0', name,
                )
                return None
            stock_conc = stock_solutes[name].concentration
            factors.append(stock_conc / target_solute.concentration)

        if len(factors) > 1 and not np.allclose(factors, factors[0], rtol=1e-3):
            logging.error(
                'DilutionToConcentrationFormulation: target solutes imply inconsistent '
                'dilution factors %s — all specified solutes must be diluted by the same factor',
                factors,
            )
            return None

        return factors[0]

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        factor = self._compute_factor_from_target()
        if factor is None:
            self._failure_reason = 'invalid_target'
            self._formulation_results = [], [], False
            return self._formulation_results
        return self._formulate_with_factor(layout, factor)


class SoluteFormulation(Formulation):
    """Subclass of Formulation. Solutes are specified; remaining volume is filled with diluent."""

    method_name: Literal['SoluteFormulation'] = 'SoluteFormulation'
    display_name: Literal['SoluteFormulation'] = 'SoluteFormulation'
    exact_match: bool = False
    diluent: Composition = Field(default_factory=Composition)

    def formulate(self, layout: LHBedLayout) -> Tuple[List[float], List[Well], bool]:
        stripped = self.target_composition.stripped()
        needed = self.target_volume + self.Extra_Volume

        # No solutes (including null composition): entire volume is diluent.
        if not stripped.solutes:
            diluent_well = next(
                (w for w in self._available_wells(layout)
                 if w.composition == self.diluent
                 and w.volume >= needed + layout.racks[w.rack_id].min_volume),
                None,
            )
            if diluent_well is None:
                logging.error(
                    'Null composition: diluent (%s) not available with sufficient volume (%.3f mL)',
                    self.diluent, needed,
                )
                self._failure_reason = 'insufficient_volume'
                self._formulation_results = [], [], False
                return self._formulation_results
            self._formulation_results = [needed], [diluent_well], True
            logging.info('SoluteFormulation: null composition, using diluent well: %s', diluent_well)
            return self._formulation_results

        # Pass 0: check if the required solute composition already exists in a single well.
        # Uses stripped to avoid zero-concentration solutes falsely matching pure-solvent wells.
        if stripped.solutes:
            for well in self._available_wells(layout):
                rack_min = layout.racks[well.rack_id].min_volume
                if well.volume >= needed + rack_min and _solutes_match(stripped, well.composition):
                    self._formulation_results = [needed], [well], True
                    logging.info('SoluteFormulation: existing well satisfies solute requirements: %s', well)
                    return self._formulation_results

        # Mixing path: solutes + diluent top-up, always ≥2 transfers.
        inflated = self._inflated_target(layout)

        # Only admit wells whose solutes are all within the union of target
        # solutes and diluent solutes. Extra solvents are acceptable; extra
        # solutes (e.g. DPPC in an isopropanol stock) would contaminate the
        # sample silently because the NNLS system only sees target components.
        allowed_solute_names = (
            {s.name for s in stripped.solutes}
            | {s.name for s in self.diluent.solutes}
        )
        solver_wells = [
            w for w in self._available_wells(layout)
            if all(s.name in allowed_solute_names for s in w.composition.solutes)
        ]

        # Solutes only — solvents are the diluent, handled by the top-up below.
        # Including solvents in the NNLS target causes the solver to draw from
        # pure-diluent wells, inflating total volume beyond inflated.
        solutes_only = stripped.model_copy(update={'solvents': []})
        result = _solve_formulation_core(
            wells=solver_wells,
            layout=layout,
            target_composition=solutes_only,
            target_volume=inflated,
            exact_match=False,
        )

        if not result['success']:
            logging.error(result['error'])
            self._failure_reason = result.get('failure_reason')
            self._formulation_results = [], [], False
            return self._formulation_results

        volumes, wells = result['volumes'], result['wells']

        diluent_volume = inflated - sum(volumes)
        if not np.isclose(diluent_volume, 0.0, atol=ZERO_VOLUME_TOLERANCE):
            if diluent_volume < 0:
                logging.error('Diluent volume less than zero; should never happen')
                self._failure_reason = 'unsolvable'
                self._formulation_results = [], [], False
                return self._formulation_results

            diluent_well = next(
                (w for w in self._available_wells(layout)
                 if w.composition == self.diluent
                 and w.volume >= diluent_volume + layout.racks[w.rack_id].min_volume),
                None,
            )
            if diluent_well is None:
                logging.error(
                    'No diluent well (%s) with sufficient volume (%.3f mL) available on bed',
                    self.diluent, diluent_volume,
                )
                self._failure_reason = 'insufficient_volume'
                self._formulation_results = [], [], False
                return self._formulation_results
            volumes += [diluent_volume]
            wells += [diluent_well]

        self._formulation_results = volumes, wells, True
        logging.info(self._formulation_results)
        return self._formulation_results
