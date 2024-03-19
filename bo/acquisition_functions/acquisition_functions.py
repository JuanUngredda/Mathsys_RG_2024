from enum import Enum, auto

from botorch.acquisition import AnalyticAcquisitionFunction, MCAcquisitionFunction, ExpectedImprovement
from torch import Tensor


class EIType(Enum):
    BOTORCH_EXPECTED_IMPROVEMENT = auto()
    MATHSYS_EXPECTED_IMPROVEMENT = auto()
    MATHSYS_MC_EXPECTED_IMPROVEMENT = auto()


def ExpectedImprovementFactory(ei_type, model, best_value):
    if ei_type is EIType.BOTORCH_EXPECTED_IMPROVEMENT:
        return ExpectedImprovement(model=model, best_f=best_value)
    elif ei_type is EIType.MATHSYS_EXPECTED_IMPROVEMENT:
        pass
    elif ei_type is EIType.MATHSYS_MC_EXPECTED_IMPROVEMENT:
        pass


class MathsysExpectedImprovement(AnalyticAcquisitionFunction):
    def forward(self, X: Tensor) -> Tensor:
        pass


class MathsysMCExpectedImprovement(MCAcquisitionFunction):
    def forward(self, X: Tensor) -> Tensor:
        pass
