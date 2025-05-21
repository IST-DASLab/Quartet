from .base import (
    NoQuantizer,
)

from .baselines import (
    UniformQuantizer,
    HalfHadamardUniformQuantizer,
    PACTQuantizer,
    LSQQuantizer,
    LSQPlusActivationQuantizer,
    LSQPlusWeightQuantizer,
)

from .quest import (
    STEQuantizer,
    ClipQuantizer,
    HalfHadamardClipQuantizer,
    HadamardClipQuantizer,
    TrustQuantizer,
    HalfHadamardTrustQuantizer,
    HadamardTrustQuantizer,
    GaussianSTEQuantizer,
    GaussianClipQuantizer,
    GaussianTrustQuantizer,
    HadamardGaussianClipQuantizer,
    HadamardGaussianTrustQuantizer,
    FP4STEQuantizer,
    FP4ClipQuantizer,
    FP4TrustQuantizer,
    HalfHadamardFP4ClipQuantizer,
    HadamardFP4ClipQuantizer,
    HalfHadamardFP4TrustQuantizer,
    HadamardFP4TrustQuantizer,
    FourEightMaskedQuantizer,
    FourEightSTEQuantizer,
    FourEightClipQuantizer,
    FourEightTrustQuantizer,
    HalfHadamardFourEightTrustQuantizer,
    HalfHadamardTwoFourTrustQuantizer,
    HadamardFourEightTrustQuantizer,
    HalfHadamardGaussianTrustQuantizer,
)

from .adabin import (
    AdaBinWeightQuantizer,
    AdaBinActivationQuantizer,
)

from .noise import (
    NoiseQuantizer,
)

from .mxfp4 import (
    AlbertTsengQuantizer,
    QuestMXFP4Quantizer,
    AlignedAlbertTsengQuantizer,
)

QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "NoiseQuantizer": NoiseQuantizer,
    "UniformQuantizer": UniformQuantizer,
    "HalfHadamardUniformQuantizer": HalfHadamardUniformQuantizer,
    "STEQuantizer": STEQuantizer,
    "ClipQuantizer": ClipQuantizer,
    "HalfHadamardClipQuantizer": HalfHadamardClipQuantizer,
    "HadamardClipQuantizer": HadamardClipQuantizer,
    "TrustQuantizer": TrustQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "HadamardTrustQuantizer": HadamardTrustQuantizer,
    "GaussianSTEQuantizer": GaussianSTEQuantizer,
    "GaussianClipQuantizer": GaussianClipQuantizer,
    "GaussianTrustQuantizer": GaussianTrustQuantizer,
    "HadamardGaussianClipQuantizer": HadamardGaussianClipQuantizer,
    "HalfHadamardGaussianTrustQuantizer": HalfHadamardGaussianTrustQuantizer,
    "HadamardGaussianTrustQuantizer": HadamardGaussianTrustQuantizer,
    "FP4STEQuantizer": FP4STEQuantizer,
    "FP4ClipQuantizer": FP4ClipQuantizer,
    "FP4TrustQuantizer": FP4TrustQuantizer,
    "HalfHadamardFP4ClipQuantizer": HalfHadamardFP4ClipQuantizer,
    "HadamardFP4ClipQuantizer": HadamardFP4ClipQuantizer,
    "HalfHadamardFP4TrustQuantizer": HalfHadamardFP4TrustQuantizer,
    "HadamardFP4TrustQuantizer": HadamardFP4TrustQuantizer,
    "FourEightMaskedQuantizer": FourEightMaskedQuantizer,
    "FourEightSTEQuantizer": FourEightSTEQuantizer,
    "FourEightClipQuantizer": FourEightClipQuantizer,
    "FourEightTrustQuantizer": FourEightTrustQuantizer,
    "HalfHadamardFourEightTrustQuantizer": HalfHadamardFourEightTrustQuantizer,
    "HalfHadamardTwoFourTrustQuantizer": HalfHadamardTwoFourTrustQuantizer,
    "HadamardFourEightTrustQuantizer": HadamardFourEightTrustQuantizer,
    "PACTQuantizer": PACTQuantizer,
    "LSQQuantizer": LSQQuantizer,
    "LSQPlusActivationQuantizer": LSQPlusActivationQuantizer,
    "LSQPlusWeightQuantizer": LSQPlusWeightQuantizer,
    "AdaBinWeightQuantizer": AdaBinWeightQuantizer,
    "AdaBinActivationQuantizer": AdaBinActivationQuantizer,
    "AlbertTsengQuantizer": AlbertTsengQuantizer,
    "QuestMXFP4Quantizer" : QuestMXFP4Quantizer,
    "AlignedAlbertTsengQuantizer": AlignedAlbertTsengQuantizer,
}
