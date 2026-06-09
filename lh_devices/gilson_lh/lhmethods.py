import asyncio
import logging

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, TYPE_CHECKING, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator, ValidationError

from lh_devices.core.bedlayout import LHBedLayout, WellLocation, Well, Solution, Composition, Solvent
from lh_devices.methods import MethodBase
from lh_devices.waste import WasteItem

from .status import MethodError, SampleStatus
from .layoutmap import LayoutWell2ZoneWell, Zone

if TYPE_CHECKING:
    from .lhinterface import LHInterface
    from .resolver import WellResolver

# ======== Gilson-LH-specific Pydantic base classes (from core/methods.py) ========

EXCLUDE_FIELDS = set(["method_name", "display_name", "complete", "method_type", "id", "tasks", "status"])


class MethodType(str, Enum):
    NONE = 'none'
    CONTAINER = 'container'
    TRANSFER = 'transfer'
    MIX = 'mix'
    INJECT = 'inject'
    PREPARE = 'prepare'
    MEASURE = 'measure'


class TaskContainer(BaseModel):
    id: str | None = None
    task: Any = Field(default_factory=dict)
    status: SampleStatus | None = None


class BaseMethod(BaseModel):
    """Base class for LH methods"""

    id: str | None = None
    tasks: list[TaskContainer] = Field(default_factory=list)
    status: SampleStatus = SampleStatus.INACTIVE
    method_name: Literal['BaseMethod'] = 'BaseMethod'
    display_name: Literal['BaseMethod'] = 'BaseMethod'
    method_type: Literal[MethodType.NONE] = MethodType.NONE

    def model_post_init(self, __context):
        if self.id is None:
            self.id = str(uuid4())

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        return None

    def new_sample_composition(self, layout: LHBedLayout) -> str:
        return ''

    def estimated_time(self, layout: LHBedLayout) -> float:
        return 0.0

    def get_methods(self, layout: LHBedLayout) -> List:
        return [self]

    def explode(self, layout: LHBedLayout) -> List:
        return self.get_methods(layout)

    def render_method(self, sample_name: str, sample_description: str,
                      layout: LHBedLayout) -> List[dict]:
        return [{}]


class UnknownMethod(BaseMethod):
    method_name: Literal['Unknown'] = 'Unknown'
    display_name: Literal['Unknown'] = 'Unknown'
    method_data: dict = Field(default_factory=dict)

    def render_method(self, sample_name: str, sample_description: str,
                      layout: LHBedLayout) -> List[dict]:
        return []


class MethodContainer(BaseMethod):
    method_type: Literal[MethodType.CONTAINER] = MethodType.CONTAINER
    method_name: Literal['MethodContainer'] = 'MethodContainer'
    display_name: Literal['MethodContainer'] = 'MethodContainer'

    def get_methods(self, layout: LHBedLayout) -> List[BaseMethod]:
        return []

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        for m in self.get_methods(layout):
            error = m.execute(layout)
            if error is not None:
                return MethodError(name=f'{self.display_name}.{error.name}', error=error.error)

    def estimated_time(self, layout: LHBedLayout) -> float:
        return sum(m.estimated_time(layout) for m in self.get_methods(layout))

    def render_method(self, sample_name: str, sample_description: str,
                      layout: LHBedLayout) -> List[dict]:
        rendered_methods = []
        for m in self.get_methods(layout):
            rendered_methods += m.render_method(sample_name=sample_name,
                                                sample_description=sample_description,
                                                layout=layout)
        return rendered_methods


MethodsType = Union[BaseMethod, MethodContainer]


def _flatten_allof_refs(schema: dict) -> None:
    """In-place: replace allOf([{$ref: X}]) with {$ref: X} in schema properties.

    Pydantic v2 wraps a $ref in allOf when the field carries extra keywords
    (e.g. a default value).  JSON Schema allows allOf([A]) ≡ A, so flattening
    is semantically correct and makes the frontend '$ref' in prop check work.
    """
    for prop in schema.get('properties', {}).values():
        if (isinstance(prop, dict)
                and 'allOf' in prop
                and len(prop['allOf']) == 1
                and '$ref' in prop['allOf'][0]):
            ref = prop['allOf'][0]['$ref']
            prop.clear()
            prop['$ref'] = ref


# ======== BaseLHMethod and concrete Pydantic methods ========

WATER = Composition(solvents=[Solvent(name='H2O', fraction=1.0)])

ORIGIN = 'lh'
EXCLUDE_LH_FIELDS = ['status', 'tasks']


class LHMethodType(str, Enum):
    TRANSFER = 'transfer'
    MIX = 'mix'
    INJECT = 'inject'


class BaseLHMethod(BaseMethod):
    """Base class for LH methods"""

    method_name: Literal['BaseLHMethod'] = 'BaseLHMethod'
    display_name: Literal['BaseLHMethod'] = 'BaseLHMethod'
    method_type: Literal[MethodType.NONE] = MethodType.NONE

    class lh_method(BaseModel):
        """Base class for representation in Trilution LH sample lists"""
        SAMPLENAME: str
        SAMPLEDESCRIPTION: str
        METHODNAME: str

        def to_dict(self) -> dict:
            return self.model_dump()

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        return None

    def waste(self, layout: LHBedLayout) -> WasteItem:
        return WasteItem()

    def new_sample_composition(self, layout: LHBedLayout) -> str:
        return ''

    def resolved_composition(self, layout: LHBedLayout) -> dict | None:
        return None

    def estimated_time(self, layout: LHBedLayout) -> float:
        return 7.0 / 60.0

    def render_method(self, sample_name: str, sample_description: str,
                      layout: LHBedLayout) -> List[dict]:
        return [{ORIGIN: [dict(sample_name=sample_name,
                               sample_description=sample_description,
                               method_name=self.method_name,
                               method_data=self.model_dump(exclude=EXCLUDE_LH_FIELDS))]}]

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[dict]:
        return [{}]


class SetWellID(BaseMethod):
    """Sets an Inferred Well Location ID for future use"""

    well: WellLocation = field(default_factory=WellLocation)
    well_id: str | None = None
    method_type: Literal[MethodType.PREPARE] = MethodType.PREPARE

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        well, _ = layout.get_well_and_rack(self.well.rack_id, self.well.well_number)
        well.id = self.well_id


class InjectMethod(BaseLHMethod):
    method_name: Literal['InjectMethod'] = 'InjectMethod'
    display_name: Literal['InjectMethod'] = 'InjectMethod'
    method_type: Literal[MethodType.INJECT] = MethodType.INJECT
    Source: WellLocation = field(default_factory=WellLocation)
    Volume: float = 1.0

    def new_sample_composition(self, layout: LHBedLayout) -> str:
        source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
        return repr(source_well.composition)

    def resolved_composition(self, layout: LHBedLayout) -> dict | None:
        try:
            source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
            return source_well.composition.model_dump()
        except Exception:
            return None

    @property
    def sample_volume(self):
        return self.Volume

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
        if self.sample_volume > source_well.volume:
            return MethodError(name=self.display_name,
                               error=f"Injection of volume {self.sample_volume} requested but well {source_well.well_number} in {source_well.rack_id} rack contains only {source_well.volume}")
        source_well.volume -= self.sample_volume


class MixMethod(BaseLHMethod):
    method_name: str = 'MixMethod'
    display_name: str = 'MixMethod'
    method_type: Literal[MethodType.MIX] = MethodType.MIX
    Target: WellLocation = field(default_factory=WellLocation)
    Volume: float = 1.0

    def new_sample_composition(self, layout: LHBedLayout) -> str:
        target_well, _ = layout.get_well_and_rack(self.Target.rack_id, self.Target.well_number)
        return repr(target_well.composition)

    @property
    def extra_volume(self):
        return 0.0

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        target_well, _ = layout.get_well_and_rack(self.Target.rack_id, self.Target.well_number)
        required_volume = self.Volume + self.extra_volume
        if required_volume > target_well.volume:
            return MethodError(name=self.display_name,
                               error=f"Mix with volume {required_volume} requested but well {target_well.well_number} in {target_well.rack_id} rack contains only {target_well.volume}")
        target_well.volume -= self.extra_volume


class TransferMethod(BaseLHMethod):
    method_name: str = 'TransferMethod'
    display_name: str = 'TransferMethod'
    method_type: Literal[MethodType.TRANSFER] = MethodType.TRANSFER
    Source: WellLocation = field(default_factory=WellLocation)
    Target: WellLocation = field(default_factory=WellLocation)
    Volume: float = 1.0

    def new_sample_composition(self, layout: LHBedLayout) -> str:
        source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
        return repr(source_well.composition)

    @property
    def transfer_volume(self):
        return self.Volume

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
        target_well, target_rack = layout.get_well_and_rack(self.Target.rack_id, self.Target.well_number)
        if self.transfer_volume > source_well.volume:
            return MethodError(name=self.display_name,
                               error=f"Well {source_well.well_number} in {source_well.rack_id} rack contains {source_well.volume} but needs {self.transfer_volume}")
        source_well.volume -= self.transfer_volume
        if (target_well.volume + self.Volume) > target_rack.max_volume:
            return MethodError(name=self.display_name,
                               error=f"Total volume {target_well.volume + self.Volume} from existing volume {target_well.volume} and transfer volume {self.Volume} exceeds rack maximum volume {target_rack.max_volume}")
        target_well.mix_with(self.Volume, source_well.composition)


class TransferWithRinse(TransferMethod):
    """Transfer with rinse"""

    method_name: Literal['NCNR_TransferWithRinse'] = 'NCNR_TransferWithRinse'
    display_name: Literal['Transfer With Rinse'] = 'Transfer With Rinse'

    Flow_Rate: float = 2.5
    Aspirate_Flow_Rate: float = 2.0
    Extra_Volume: float = 0.1
    Outside_Rinse_Volume: float = 0.5
    Inside_Rinse_Volume: float = 0.5
    Air_Gap: float = 0.1
    Use_Liquid_Level_Detection: bool = True

    class lh_method(BaseLHMethod.lh_method):
        Source_Zone: Zone
        Source_Well: str
        Volume: str
        Flow_Rate: str
        Aspirate_Flow_Rate: str
        Extra_Volume: str
        Outside_Rinse_Volume: str
        Inside_Rinse_Volume: str
        Air_Gap: str
        Use_Liquid_Level_Detection: str
        Target_Zone: Zone
        Target_Well: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        self.Source = resolver.resolve(self.Source)
        source_zone, source_well = LayoutWell2ZoneWell(self.Source.rack_id, self.Source.well_number)
        self.Target = resolver.resolve(self.Target)
        target_zone, target_well = LayoutWell2ZoneWell(self.Target.rack_id, self.Target.well_number)
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME=self.method_name,
            Source_Zone=source_zone,
            Source_Well=source_well,
            Volume=f'{self.Volume}',
            Flow_Rate=f'{self.Flow_Rate}',
            Aspirate_Flow_Rate=f'{self.Aspirate_Flow_Rate}',
            Extra_Volume=f'{self.Extra_Volume}',
            Outside_Rinse_Volume=f'{self.Outside_Rinse_Volume}',
            Inside_Rinse_Volume=f'{self.Inside_Rinse_Volume}',
            Air_Gap=f'{self.Air_Gap}',
            Use_Liquid_Level_Detection=f'{self.Use_Liquid_Level_Detection}',
            Target_Zone=target_zone,
            Target_Well=target_well
        ).to_dict()]

    @property
    def transfer_volume(self):
        return self.Volume + self.Extra_Volume

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        rinse_time = 23.0 / 60.0
        return self.Volume / self.Flow_Rate + self.Volume / self.Aspirate_Flow_Rate + self.Air_Gap / 0.3 + base_time + rinse_time

    def execute(self, layout):
        layout.carrier_well.volume -= (self.Outside_Rinse_Volume + self.Inside_Rinse_Volume)
        return super().execute(layout)

    def waste(self, layout: LHBedLayout) -> WasteItem:
        inferred_source_well = layout.infer_location(self.Source)
        if inferred_source_well.well_number is not None:
            source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
            source_composition = source_well.composition
        else:
            source_composition = self.Source.expected_composition
        new_waste = WasteItem()
        new_waste.mix_with(volume=self.Extra_Volume, composition=source_composition)
        new_waste.mix_with(volume=self.Outside_Rinse_Volume + self.Inside_Rinse_Volume, composition=layout.carrier_well.composition)
        return new_waste


class MixWithRinse(MixMethod):
    """Inject with rinse"""
    Flow_Rate: float = 2.5
    Aspirate_Flow_Rate: float = 2.0
    Extra_Volume: float = 0.1
    Outside_Rinse_Volume: float = 0.5
    Inside_Rinse_Volume: float = 0.5
    Air_Gap: float = 0.1
    Repeats: int = 3
    Use_Liquid_Level_Detection: bool = True
    display_name: Literal['Mix With Rinse'] = 'Mix With Rinse'
    method_name: Literal['NCNR_MixWithRinse'] = 'NCNR_MixWithRinse'

    class lh_method(BaseLHMethod.lh_method):
        Volume: str
        Flow_Rate: str
        Aspirate_Flow_Rate: str
        Extra_Volume: str
        Outside_Rinse_Volume: str
        Inside_Rinse_Volume: str
        Air_Gap: str
        Use_Liquid_Level_Detection: str
        Repeats: str
        Target_Zone: Zone
        Target_Well: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        self.Target = resolver.resolve(self.Target)
        target_zone, target_well = LayoutWell2ZoneWell(self.Target.rack_id, self.Target.well_number)
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME=self.method_name,
            Volume=f'{self.Volume}',
            Flow_Rate=f'{self.Flow_Rate}',
            Aspirate_Flow_Rate=f'{self.Aspirate_Flow_Rate}',
            Extra_Volume=f'{self.Extra_Volume}',
            Outside_Rinse_Volume=f'{self.Outside_Rinse_Volume}',
            Inside_Rinse_Volume=f'{self.Inside_Rinse_Volume}',
            Air_Gap=f'{self.Air_Gap}',
            Use_Liquid_Level_Detection=f'{self.Use_Liquid_Level_Detection}',
            Repeats=f'{self.Repeats}',
            Target_Zone=target_zone,
            Target_Well=target_well
        ).to_dict()]

    @property
    def extra_volume(self):
        return self.Extra_Volume

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        target_well, _ = layout.get_well_and_rack(self.Target.rack_id, self.Target.well_number)
        if self.Volume > target_well.volume:
            return MethodError(name=self.display_name,
                               error=f"Mix with volume {self.Volume} requested but well {target_well.well_number} in {target_well.rack_id} rack contains only {target_well.volume}")
        target_well.volume -= self.Extra_Volume
        layout.carrier_well.volume -= (self.Outside_Rinse_Volume + self.Inside_Rinse_Volume)

    def waste(self, layout: LHBedLayout) -> WasteItem:
        inferred_target_well = layout.infer_location(self.Target)
        if inferred_target_well.well_number is not None:
            target_well, _ = layout.get_well_and_rack(inferred_target_well.rack_id, inferred_target_well.well_number)
            target_composition = target_well.composition
        else:
            target_composition = self.Target.expected_composition
        new_waste = WasteItem()
        new_waste.mix_with(volume=self.Extra_Volume, composition=target_composition)
        new_waste.mix_with(volume=self.Outside_Rinse_Volume + self.Inside_Rinse_Volume, composition=layout.carrier_well.composition)
        return new_waste

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        rinse_time = 23.0 / 60.0
        return self.Repeats * (self.Volume / self.Flow_Rate + self.Volume / self.Aspirate_Flow_Rate) + self.Air_Gap / 0.3 + base_time + rinse_time


class InjectWithRinse(InjectMethod):
    """Inject with rinse"""
    Aspirate_Flow_Rate: float = 2.0
    Flow_Rate: float = 2.5
    Extra_Volume: float = 0.1
    Outside_Rinse_Volume: float = 0.5
    Air_Gap: float = 0.1
    Use_Liquid_Level_Detection: bool = True
    display_name: Literal['Inject With Rinse'] = 'Inject With Rinse'
    method_name: Literal['NCNR_InjectWithRinse'] = 'NCNR_InjectWithRinse'

    class lh_method(BaseLHMethod.lh_method):
        Source_Zone: Zone
        Source_Well: str
        Volume: str
        Aspirate_Flow_Rate: str
        Flow_Rate: str
        Extra_Volume: str
        Outside_Rinse_Volume: str
        Air_Gap: str
        Use_Liquid_Level_Detection: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        self.Source = resolver.resolve(self.Source)
        source_zone, source_well = LayoutWell2ZoneWell(self.Source.rack_id, self.Source.well_number)
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME=self.method_name,
            Source_Zone=source_zone,
            Source_Well=source_well,
            Volume=f'{self.Volume}',
            Aspirate_Flow_Rate=f'{self.Aspirate_Flow_Rate}',
            Flow_Rate=f'{self.Flow_Rate}',
            Extra_Volume=f'{self.Extra_Volume}',
            Outside_Rinse_Volume=f'{self.Outside_Rinse_Volume}',
            Air_Gap=f'{self.Air_Gap}',
            Use_Liquid_Level_Detection=f'{self.Use_Liquid_Level_Detection}'
        ).to_dict()]

    @property
    def sample_volume(self):
        return self.Volume + self.Extra_Volume

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        rinse_time = 23.0 / 60.0
        return self.Volume / self.Aspirate_Flow_Rate + self.Volume / self.Flow_Rate + self.Air_Gap / 0.3 + base_time + rinse_time

    def execute(self, layout):
        layout.carrier_well.volume -= self.Outside_Rinse_Volume + 0.5
        return super().execute(layout)

    def waste(self, layout: LHBedLayout) -> WasteItem:
        inferred_source_well = layout.infer_location(self.Source)
        if inferred_source_well.well_number is not None:
            source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
            source_composition = source_well.composition
        else:
            source_composition = self.Source.expected_composition
        new_waste = WasteItem()
        new_waste.mix_with(volume=self.Volume + self.Extra_Volume, composition=source_composition)
        new_waste.mix_with(volume=self.Outside_Rinse_Volume + 0.5, composition=layout.carrier_well.composition)
        return new_waste


class Sleep(BaseLHMethod):
    """Sleep"""
    Time: float = 1.0
    display_name: Literal['Sleep'] = 'Sleep'
    method_name: Literal['NCNR_Sleep'] = 'NCNR_Sleep'

    class lh_method(BaseLHMethod.lh_method):
        Time: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME=self.method_name,
            Time=f'{self.Time}'
        ).to_dict()]

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        return float(self.Time) + base_time


class Prime(BaseLHMethod):
    """Prime"""
    Volume: float = 10.0
    Repeats: int = 1
    display_name: Literal['Prime'] = 'Prime'
    method_name: Literal['NCNR_Prime'] = 'NCNR_Prime'

    class lh_method(BaseLHMethod.lh_method):
        Volume: str
        Repeats: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME=self.method_name,
            Volume=f'{self.Volume}',
            Repeats=f'{self.Repeats:0.0f}'
        ).to_dict()]

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        flow_rate = 10.0
        return 2 * float(self.Repeats) * float(self.Volume) / flow_rate + base_time

    def execute(self, layout):
        layout.carrier_well.volume -= self.Volume * self.Repeats
        return super().execute(layout)

    def waste(self, layout: LHBedLayout) -> WasteItem:
        return WasteItem(volume=self.Volume * self.Repeats, composition=layout.carrier_well.composition)


class ROADMAP_QCMD_LoadLoop(InjectMethod):
    """Load injection system loop"""
    Aspirate_Flow_Rate: float = 2.5
    Flow_Rate: float = 2.5
    Outside_Rinse_Volume: float = 0.5
    Extra_Volume: float = 0.1
    Air_Gap: float = 0.1
    Use_Liquid_Level_Detection: bool = True
    display_name: Literal['Load Injection System Loop'] = 'Load Injection System Loop'
    method_name: Literal['ROADMAP_QCMD_LoadLoop'] = 'ROADMAP_QCMD_LoadLoop'

    class lh_method(BaseLHMethod.lh_method):
        Source_Zone: Zone
        Source_Well: str
        Volume: str
        Aspirate_Flow_Rate: str
        Flow_Rate: str
        Outside_Rinse_Volume: str
        Extra_Volume: str
        Air_Gap: str
        Use_Liquid_Level_Detection: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        self.Source = resolver.resolve(self.Source)
        source_zone, source_well = LayoutWell2ZoneWell(self.Source.rack_id, self.Source.well_number)
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME='ROADMAP_QCMD_LoadLoop',
            Source_Zone=source_zone,
            Source_Well=source_well,
            Volume=f'{self.Volume}',
            Aspirate_Flow_Rate=f'{self.Aspirate_Flow_Rate}',
            Flow_Rate=f'{self.Flow_Rate}',
            Outside_Rinse_Volume=f'{self.Outside_Rinse_Volume}',
            Extra_Volume=f'{self.Extra_Volume}',
            Air_Gap=f'{self.Air_Gap}',
            Use_Liquid_Level_Detection=f'{self.Use_Liquid_Level_Detection}',
        ).to_dict()]

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        rinse_time = 23.0 / 60.0
        return self.Volume / self.Aspirate_Flow_Rate + self.Volume / self.Flow_Rate + self.Air_Gap / 0.3 + base_time + rinse_time

    def execute(self, layout):
        layout.carrier_well.volume -= self.Outside_Rinse_Volume + 0.5
        return super().execute(layout)

    def waste(self, layout: LHBedLayout) -> WasteItem:
        inferred_source_well = layout.infer_location(self.Source)
        if inferred_source_well.well_number is not None:
            source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
            source_composition = source_well.composition
        else:
            source_composition = self.Source.expected_composition
        new_waste = WasteItem()
        new_waste.mix_with(volume=self.Volume + self.Extra_Volume, composition=source_composition)
        new_waste.mix_with(volume=self.Outside_Rinse_Volume + 0.5, composition=layout.carrier_well.composition)
        return new_waste

    @property
    def sample_volume(self):
        return self.Volume + self.Extra_Volume


class ROADMAP_QCMD_DirectInject(InjectMethod):
    """Direct inject with rinse"""
    Aspirate_Flow_Rate: float = 2.5
    Load_Flow_Rate: float = 2.5
    Injection_Flow_Rate: float = 1.0
    Outside_Rinse_Volume: float = 0.5
    Extra_Volume: float = 0.1
    Air_Gap: float = 0.1
    Use_Liquid_Level_Detection: bool = True
    Use_Bubble_Sensors: bool = True
    display_name: Literal['Direct Inject'] = 'Direct Inject'
    method_name: Literal['ROADMAP_QCMD_DirectInject'] = 'ROADMAP_QCMD_DirectInject'

    class lh_method(BaseLHMethod.lh_method):
        Source_Zone: Zone
        Source_Well: str
        Volume: str
        Aspirate_Flow_Rate: str
        Load_Flow_Rate: str
        Injection_Flow_Rate: str
        Outside_Rinse_Volume: str
        Extra_Volume: str
        Air_Gap: str
        Use_Liquid_Level_Detection: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[dict]:
        self.Source = resolver.resolve(self.Source)
        source_zone, source_well_number = LayoutWell2ZoneWell(self.Source.rack_id, self.Source.well_number)
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME='ROADMAP_QCMD_DirectInject_BubbleSensor' if self.Use_Bubble_Sensors else 'ROADMAP_QCMD_DirectInject',
            Source_Zone=source_zone,
            Source_Well=source_well_number,
            Volume=f'{self.Volume}',
            Aspirate_Flow_Rate=f'{self.Aspirate_Flow_Rate}',
            Load_Flow_Rate=f'{self.Load_Flow_Rate}',
            Injection_Flow_Rate=f'{self.Injection_Flow_Rate}',
            Outside_Rinse_Volume=f'{self.Outside_Rinse_Volume}',
            Extra_Volume=f'{self.Extra_Volume}',
            Air_Gap=f'{self.Air_Gap}',
            Use_Liquid_Level_Detection=f'{self.Use_Liquid_Level_Detection}',
        ).to_dict()]

    def estimated_time(self, layout: LHBedLayout) -> float:
        base_time = super().estimated_time(layout)
        rinse_time = 23.0 / 60.0
        return self.Volume / self.Aspirate_Flow_Rate + self.Volume / self.Injection_Flow_Rate + self.Air_Gap / 0.3 + base_time + rinse_time

    def execute(self, layout):
        layout.carrier_well.volume -= self.Outside_Rinse_Volume + 0.5
        return super().execute(layout)

    def waste(self, layout: LHBedLayout) -> WasteItem:
        inferred_source_well = layout.infer_location(self.Source)
        if inferred_source_well.well_number is not None:
            source_well, _ = layout.get_well_and_rack(self.Source.rack_id, self.Source.well_number)
            source_composition = source_well.composition
        else:
            source_composition = self.Source.expected_composition
        new_waste = WasteItem()
        new_waste.mix_with(volume=self.Volume + self.Extra_Volume, composition=source_composition)
        new_waste.mix_with(volume=self.Outside_Rinse_Volume + 0.5, composition=layout.carrier_well.composition)
        return new_waste

    @property
    def sample_volume(self):
        return self.Volume + self.Extra_Volume


class ROADMAP_DirectInjectPrime(BaseLHMethod):
    """Flush direct injection line with carrier liquid"""
    Volume: float = 3.0
    Flow_Rate: float = 3.0
    display_name: Literal['ROADMAP Direct Inject Prime'] = 'ROADMAP Direct Inject Prime'
    method_name: Literal['ROADMAP_DirectInjectPrime'] = 'ROADMAP_DirectInjectPrime'
    method_type: Literal[MethodType.NONE] = MethodType.NONE

    class lh_method(BaseLHMethod.lh_method):
        Volume: str
        Flow_Rate: str

    def render_lh_method(self, sample_name: str, sample_description: str,
                         resolver: 'WellResolver') -> List[BaseLHMethod.lh_method]:
        return [self.lh_method(
            SAMPLENAME=sample_name,
            SAMPLEDESCRIPTION=sample_description,
            METHODNAME=self.method_name,
            Volume=f'{self.Volume}',
            Flow_Rate=f'{self.Flow_Rate}',
        ).to_dict()]

    def estimated_time(self, layout: LHBedLayout) -> float:
        return self.Volume / self.Flow_Rate

    def execute(self, layout: LHBedLayout) -> MethodError | None:
        layout.carrier_well.volume -= self.Volume
        return None

    def waste(self, layout: LHBedLayout) -> WasteItem:
        return WasteItem(volume=self.Volume, composition=layout.carrier_well.composition)


# Local class registry — used by LHMethodCluster for deserialization
_LH_METHOD_CLASSES: dict[str, type] = {
    cls.model_fields['method_name'].default: cls
    for cls in [TransferWithRinse, MixWithRinse, InjectWithRinse, Sleep, Prime,
                ROADMAP_QCMD_LoadLoop, ROADMAP_QCMD_DirectInject, ROADMAP_DirectInjectPrime]
}


class LHMethodCluster(BaseLHMethod):

    method_name: Literal['LHMethodCluster'] = 'LHMethodCluster'
    display_name: Literal['LHMethodCluster'] = 'LHMethodCluster'
    method_type: MethodType = MethodType.PREPARE
    methods: list = field(default_factory=list)

    @validator('methods')
    def validate_methods(cls, v):
        if not isinstance(v, list):
            raise ValueError(f"{v} must be a list")
        for i, iv in enumerate(v):
            if isinstance(iv, dict):
                method_cls = _LH_METHOD_CLASSES.get(iv.get('method_name', ''))
                if method_cls is not None:
                    try:
                        v[i] = method_cls.model_validate(iv)
                    except ValidationError:
                        logging.warning(f'Attempted to process unknown method with data {iv}')
                        v[i] = UnknownMethod(method_data=iv)
                else:
                    v[i] = UnknownMethod(method_data=iv)
            else:
                if not isinstance(iv, BaseMethod):
                    raise ValueError(f"{iv} must be derived from BaseMethod")
        return v

    def explode(self, layout: LHBedLayout):
        methods = []
        for m in self.methods:
            methods += m.explode(layout)
        return methods

    def render_method(self, sample_name: str, sample_description: str, layout: LHBedLayout) -> List[dict]:
        return [{ORIGIN: [dict(sample_name=sample_name,
                               sample_description=sample_description,
                               method_name=m.method_name,
                               method_data=m.model_dump(exclude=EXCLUDE_FIELDS))
                          for m in self.methods]}]

    def estimated_time(self, layout: LHBedLayout) -> float:
        return sum(m.estimated_time(layout) for m in self.methods)

    def get_methods(self, layout: LHBedLayout) -> list[MethodsType]:
        return self.methods


# ======== GilsonLHMethod: lh_devices MethodBase wrappers ========

class GilsonLHMethod(MethodBase):
    """Wraps a BaseLHMethod Pydantic class into the lh_devices MethodBase framework.

    Subclasses set _lh_method_class to the Pydantic class to wrap and override
    MethodDefinition.name to match the Pydantic method_name.
    """
    _lh_method_class: ClassVar[type[BaseLHMethod]]

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'GilsonLHMethod'

    def __init__(self, lh_iface: 'LHInterface') -> None:
        super().__init__(devices=[lh_iface])
        self.lh_iface = lh_iface

    @classmethod
    def get_pydantic_schema(cls, display: bool = True, origin: str = 'lh') -> dict:
        """Returns parameter schema from the underlying Pydantic class."""
        m = cls._lh_method_class
        EXCLUDE = {'status', 'tasks', 'id', 'method_name', 'display_name', 'method_type'}
        schema = m.model_json_schema(mode='serialization')
        _flatten_allof_refs(schema)
        return {
            'fields': [f for f in m.model_fields if f not in EXCLUDE],
            'display': display,
            'display_name': m.model_fields['display_name'].default,
            'method_type': m.model_fields['method_type'].default,
            'origin': origin,
            'schema': schema,
        }

    async def _run_job(self, job, lh_methods: list) -> dict:
        """Activate a job and block until Trilution completes it.

        Returns dict with 'waste' list and optional 'resolved_composition'.
        Raises Exception on failure (caught by MethodBase.start() as a plain error).
        """
        from .job import ValidationStatus, ResultStatus

        done = asyncio.Event()
        error_holder: dict = {}
        captured_job: list = []

        async def _on_validation(val_job, validation_status, *args, **kwargs) -> None:
            if val_job.id == job.id and validation_status != ValidationStatus.SUCCESS:
                detail = val_job.validation
                error_holder['error'] = f'Validation failed: {detail}'
                done.set()

        async def _on_result(result_job, *args, **kwargs) -> None:
            if result_job.id != job.id:
                return
            status = result_job.get_result_status()
            if status in (ResultStatus.FAIL, ResultStatus.SUCCESS):
                if status == ResultStatus.FAIL:
                    error_holder['error'] = 'LH job failed'
                captured_job.append(result_job)
                done.set()

        self.lh_iface.validation_callbacks.append(_on_validation)
        self.lh_iface.results_callbacks.append(_on_result)
        try:
            await self.lh_iface.activate_job(job)
            cancel_task = asyncio.ensure_future(self.lh_iface._job_cancelled.wait())
            done_task = asyncio.ensure_future(done.wait())
            _, pending = await asyncio.wait(
                [done_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            if self.lh_iface._job_cancelled.is_set():
                raise Exception('Job cancelled by operator')
            if 'error' in error_holder:
                await self.lh_iface.throw_error(error_holder['error'])
                raise Exception(error_holder['error'])
            # Collect waste BEFORE execute (pre-mutation layout state)
            waste_items = []
            for m in lh_methods:
                try:
                    waste_items.append(m.waste(self.lh_iface.layout).model_dump())
                except Exception:
                    pass
            # Mutate layout
            completed_job = captured_job[0] if captured_job else job
            completed_job.execute_methods(self.lh_iface.layout)
            await self.lh_iface.trigger_layout_update()
            # Resolved composition post-mutation
            result: dict = {'waste': waste_items}
            for m in lh_methods:
                try:
                    rc = m.resolved_composition(self.lh_iface.layout)
                    if rc is not None:
                        result['resolved_composition'] = rc
                        break
                except Exception:
                    pass
            return result
        finally:
            if _on_validation in self.lh_iface.validation_callbacks:
                self.lh_iface.validation_callbacks.remove(_on_validation)
            if _on_result in self.lh_iface.results_callbacks:
                self.lh_iface.results_callbacks.remove(_on_result)
            await self.lh_iface.deactivate()

    async def run(self, sample_id: str = '', task_id: str | None = None, **kwargs) -> dict:
        layout = self.lh_iface.layout
        if layout is None:
            raise Exception('LH layout not loaded')
        self.reserve_all()
        try:
            lh_method = self._lh_method_class(**kwargs)
            flat_methods = lh_method.explode(layout)
            from .lhinterface import LHJob
            job = LHJob(id=task_id or str(uuid4()))
            job.setup_method_data(sample_id, '', flat_methods, layout)
            return await self._run_job(job, flat_methods)
        finally:
            self.release_all()


class GilsonTransferWithRinse(GilsonLHMethod):
    _lh_method_class = TransferWithRinse

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'NCNR_TransferWithRinse'


class GilsonMixWithRinse(GilsonLHMethod):
    _lh_method_class = MixWithRinse

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'NCNR_MixWithRinse'


class GilsonInjectWithRinse(GilsonLHMethod):
    _lh_method_class = InjectWithRinse

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'NCNR_InjectWithRinse'


class GilsonSleep(GilsonLHMethod):
    _lh_method_class = Sleep

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'NCNR_Sleep'


class GilsonPrime(GilsonLHMethod):
    _lh_method_class = Prime

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'NCNR_Prime'
        Volume: str | float = 10.0
        Repeats: str | int = 1


class GilsonQCMDLoadLoop(GilsonLHMethod):
    _lh_method_class = ROADMAP_QCMD_LoadLoop

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'ROADMAP_QCMD_LoadLoop'


class GilsonQCMDDirectInject(GilsonLHMethod):
    _lh_method_class = ROADMAP_QCMD_DirectInject

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'ROADMAP_QCMD_DirectInject'


class GilsonDirectInjectPrime(GilsonLHMethod):
    _lh_method_class = ROADMAP_DirectInjectPrime

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'ROADMAP_DirectInjectPrime'


class GilsonFormulation(GilsonLHMethod):
    """MethodBase wrapper for Formulation — lazy-imports to avoid circular."""

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'Formulation'

    @classmethod
    def get_pydantic_schema(cls, display: bool = True, origin: str = 'lh') -> dict:
        from .formulation import Formulation
        m = Formulation
        EXCLUDE = {'status', 'tasks', 'id', 'method_name', 'display_name', 'method_type'}
        schema = m.model_json_schema(mode='serialization')
        _flatten_allof_refs(schema)
        return {
            'fields': [f for f in m.model_fields if f not in EXCLUDE],
            'display': display,
            'display_name': m.model_fields['display_name'].default,
            'method_type': m.model_fields['method_type'].default,
            'origin': origin,
            'schema': schema,
        }

    async def run(self, sample_id: str = '', task_id: str | None = None, **kwargs) -> dict:
        from .formulation import Formulation
        layout = self.lh_iface.layout
        if layout is None:
            raise Exception('LH layout not loaded')
        self.reserve_all()
        try:
            lh_method = Formulation(**kwargs)
            clusters = lh_method.get_methods(layout, sample_id=sample_id or None)
            flat_methods = [m2 for cluster in clusters for m2 in cluster.explode(layout)]
            if not flat_methods:
                _, _, success = lh_method.get_formulation_results(layout)
                if not success:
                    raise RuntimeError('Formulation failed: no valid source wells found')
                return {}
            from .lhinterface import LHJob
            job = LHJob(id=task_id or str(uuid4()))
            job.setup_method_data(sample_id, '', flat_methods, layout)
            return await self._run_job(job, flat_methods)
        finally:
            self.release_all()


class GilsonSoluteFormulation(GilsonLHMethod):
    """MethodBase wrapper for SoluteFormulation — lazy-imports to avoid circular."""

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        name: str = 'SoluteFormulation'

    @classmethod
    def get_pydantic_schema(cls, display: bool = True, origin: str = 'lh') -> dict:
        from .formulation import SoluteFormulation
        m = SoluteFormulation
        EXCLUDE = {'status', 'tasks', 'id', 'method_name', 'display_name', 'method_type'}
        schema = m.model_json_schema(mode='serialization')
        _flatten_allof_refs(schema)
        return {
            'fields': [f for f in m.model_fields if f not in EXCLUDE],
            'display': display,
            'display_name': m.model_fields['display_name'].default,
            'method_type': m.model_fields['method_type'].default,
            'origin': origin,
            'schema': schema,
        }

    async def run(self, sample_id: str = '', task_id: str | None = None, **kwargs) -> dict:
        from .formulation import SoluteFormulation
        layout = self.lh_iface.layout
        if layout is None:
            raise Exception('LH layout not loaded')
        self.reserve_all()
        try:
            lh_method = SoluteFormulation(**kwargs)
            clusters = lh_method.get_methods(layout, sample_id=sample_id or None)
            flat_methods = [m2 for cluster in clusters for m2 in cluster.explode(layout)]
            if not flat_methods:
                _, _, success = lh_method.get_formulation_results(layout)
                if not success:
                    raise RuntimeError('SoluteFormulation failed: no valid source wells found')
                return {}
            from .lhinterface import LHJob
            job = LHJob(id=task_id or str(uuid4()))
            job.setup_method_data(sample_id, '', flat_methods, layout)
            return await self._run_job(job, flat_methods)
        finally:
            self.release_all()
