from enum import Enum, auto
from typing import Optional, Union, Any

import torch
from botorch.acquisition import ExpectedImprovement, \
    qExpectedImprovement, MCAcquisitionObjective
from botorch.acquisition.analytic import _ei_helper
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from torch import Tensor


class EIType(Enum):
    BOTORCH_EXPECTED_IMPROVEMENT = auto()
    BOTORCH_MC_EXPECTED_IMPROVEMENT = auto()
    MATHSYS_EXPECTED_IMPROVEMENT = auto()
    MATHSYS_MC_EXPECTED_IMPROVEMENT = auto()


def ExpectedImprovementFactory(ei_type, model, best_value):
    if ei_type is EIType.BOTORCH_EXPECTED_IMPROVEMENT:
        return ExpectedImprovement(model=model, best_f=best_value)
    elif ei_type is EIType.BOTORCH_MC_EXPECTED_IMPROVEMENT:
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1000]))
        return qExpectedImprovement(model=model, best_f=best_value, sampler=qmc_sampler)
    elif ei_type is EIType.MATHSYS_EXPECTED_IMPROVEMENT:
        return MathsysExpectedImprovement(model=model, best_f=best_value)
    elif ei_type is EIType.MATHSYS_MC_EXPECTED_IMPROVEMENT:
        return MathsysMCExpectedImprovement(model=model, best_f=best_value)


class MathsysExpectedImprovement(ExpectedImprovement):

    def __init__(self, model: Model, best_f: Union[float, Tensor],
                 posterior_transform: Optional[PosteriorTransform] = None, maximize: bool = True, **kwargs):
        super().__init__(model, best_f, posterior_transform, maximize, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        posterior = self.model.posterior(X)
        mu = posterior.mean
        sigma = torch.sqrt(posterior.variance)
        Z = (mu - self.best_f) / sigma
        ei_value = sigma * _ei_helper(Z)
        return ei_value.reshape(-1)


class MathsysMCExpectedImprovement(qExpectedImprovement):
    def __init__(self, model: Model, best_f: Union[float, Tensor], sampler: Optional[MCSampler] = None,
                 objective: Optional[MCAcquisitionObjective] = None,
                 posterior_transform: Optional[PosteriorTransform] = None, X_pending: Optional[Tensor] = None,
                 **kwargs: Any) -> None:
        super().__init__(model, best_f, sampler, objective, posterior_transform, X_pending, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        pass
