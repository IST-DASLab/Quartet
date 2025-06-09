from dataclasses import dataclass, field
from enum import Enum


class QuartetDtype(Enum):
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"

@dataclass
class QuartetConfig:
    forward_dtype: QuartetDtype = QuartetDtype.MXFP4
    backward_dtype: QuartetDtype = QuartetDtype.MXFP4
    store_master_weights: bool = False
    hadamard_group_size: int = 32
    modules_to_not_convert: list[str] = field(default_factory=lambda: ["lm_head"])