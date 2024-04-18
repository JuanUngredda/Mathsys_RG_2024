from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor
from typing import Union, List
import math
import torch


class ConstrainedBranin(ConstrainedBaseTestProblem):
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = (
                X_tf[..., 1]
                - 5.1 / (4 * math.pi ** 2) * X_tf[..., 0] ** 2
                + 5 / math.pi * X_tf[..., 0]
                - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X_tf[..., 0])
        return t1 ** 2 + t2 + 10

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return 50 - (X_tf[..., 0:1] - 2.5).pow(2) - (X_tf[..., 1:2] - 7.5).pow(2)

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X)
        return torch.concat([y, c1], dim=1)

class testing_function(ConstrainedBaseTestProblem):
    _bounds = [(-2.7, 7.5)]

    def __init__(self, noise_std: Union[None, float, List[float]] = None, negate: bool = False) -> None:
        self.dim = 1
        super().__init__(noise_std, negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return torch.sin(X_tf)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return self.expensive_function(X_tf)

    def expensive_function(self, x):
        x = x - 0.55
        return torch.sin(x) + torch.sin((2.0) * x)
