"""Pure formulation solver — no zone or device dependencies.

Callers are responsible for filtering the well list before passing it in.
The gilson_lh layer provides zone-aware wrappers on top of these helpers.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls

from lh_devices.core.bedlayout import Composition, LHBedLayout, Well

ZERO_VOLUME_TOLERANCE = 1e-3


def make_target_vector(
    target_composition: Composition,
) -> Tuple[List[str], List[float], Dict[str, str]]:
    solvent_names, solvent_fractions = target_composition.get_solvent_fractions()
    solute_names, solute_concentrations, solute_units = target_composition.get_solute_concentrations()
    concentrations_with_units = {name: unit for name, unit in zip(solute_names, solute_units)}
    return solvent_names + solute_names, solvent_fractions + solute_concentrations, concentrations_with_units


def select_wells(
    wells: List[Well],
    target_names: List[str],
    exact_match: bool,
) -> Tuple[List[Well], List[str]]:
    acceptable_wells = []
    all_components: set = set()
    for well in wells:
        components = well.composition.get_solvent_names() + well.composition.get_solute_names()
        if exact_match:
            if all(cmp in target_names for cmp in components):
                acceptable_wells.append(well)
        else:
            if any(cmp in target_names for cmp in components):
                acceptable_wells.append(well)
        all_components.update(components)
    return acceptable_wells, list(all_components)


def make_source_matrix(
    source_names: List[str],
    wells: List[Well],
    source_units: Dict[str, str],
) -> Tuple[List[list], List[Well]]:
    source_matrix = []
    relevant_wells = []
    for well in wells:
        composition = well.composition
        solvent_names, solvent_fractions = composition.get_solvent_fractions()
        solute_names = composition.get_solute_names()
        col = []
        for name in source_names:
            if name in solvent_names:
                col.append(solvent_fractions[solvent_names.index(name)])
            elif name in solute_names:
                col.append(composition.solutes[solute_names.index(name)].convert_units(source_units[name]))
            else:
                col.append(0)
        if sum(col):
            source_matrix.append(col)
            relevant_wells.append(well)
    source_matrix = [list(x) for x in zip(*source_matrix)]
    return source_matrix, relevant_wells


def solve_formulation(
    wells: List[Well],
    layout: LHBedLayout,
    target_composition: Composition,
    target_volume: float,
    exact_match: bool = True,
) -> Dict[str, Any]:
    """Solve a formulation from a pre-filtered list of candidate wells.

    Returns dict with keys: success, error, volumes, wells.
    """
    target_names, target_vector, target_units = make_target_vector(target_composition)
    logging.info('target names: %s', target_names)
    logging.info('target vector: %s', target_vector)

    source_wells, source_components = select_wells(wells, target_names, exact_match)

    if not source_wells:
        return {'success': False, 'error': 'Cannot create formulation: no acceptable solutions available', 'volumes': [], 'wells': []}

    for target_name in target_names:
        if target_name not in source_components:
            return {'success': False, 'error': f'Cannot make formulation: {target_name} is missing', 'volumes': [], 'wells': []}

    source_wells_current = list(source_wells)

    while True:
        logging.info('Source wells: %s', source_wells_current)
        source_matrix, source_wells_current = make_source_matrix(target_names, source_wells_current, target_units)

        if not source_wells_current:
            return {'success': False, 'error': 'Solver failed: Ran out of source wells', 'volumes': [], 'wells': []}

        logging.info('Source matrix: %s', source_matrix)

        sol, res = nnls(source_matrix, target_vector)

        if np.isclose(res, 0.0, atol=1e-9):
            logging.info('Good residual %s, solution %s', f'{res:0.0e}', sol)
            source_well_volumes = [well.volume for well in source_wells_current]
            required_volumes = sol * target_volume

            wells_to_remove = []
            for well, source_well_volume, required_volume in zip(source_wells_current, source_well_volumes, required_volumes):
                rack_min = layout.racks[well.rack_id].min_volume
                if (required_volume + rack_min) > source_well_volume:
                    logging.warning('Well %s insufficient volume (needs %s + %s, has %s). Removing.', well, required_volume, rack_min, source_well_volume)
                    wells_to_remove.append(well)

            if not wells_to_remove:
                volumes = []
                result_wells = []
                for well, vol in zip(source_wells_current, sol):
                    if vol * target_volume > ZERO_VOLUME_TOLERANCE:
                        volumes.append(vol * target_volume)
                        result_wells.append(well)
                return {'success': True, 'error': None, 'volumes': volumes, 'wells': result_wells}
            else:
                for w in wells_to_remove:
                    if w in source_wells_current:
                        source_wells_current.remove(w)
                if not source_wells_current:
                    return {'success': False, 'error': 'Insufficient volume in source wells', 'volumes': [], 'wells': []}
        else:
            logging.warning('Bad residual %s', f'{res:0.0e}')
            return {'success': False, 'error': f'Cannot solve formulation (Residual: {res:0.0e})', 'volumes': [], 'wells': []}
