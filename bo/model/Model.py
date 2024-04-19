from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction, MCAcquisitionObjective
from botorch.models import FixedNoiseGP, ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.utils import t_batch_mode_transform
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor

# constants
device = torch.device("cpu")
dtype = torch.float64


class GPModelWrapper():
    def __init__(self):
        self.train_yvar = torch.tensor(1e-6, device=device, dtype=dtype)

    def fit(self, X, y):
        self.model = FixedNoiseGP(train_X=X,
                                  train_Y=y,
                                  train_Yvar=self.train_yvar.expand_as(y),
                                  outcome_transform=Standardize(m=1))
        return self.model

    def optimize(self):
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model


class ConstrainedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Constrained Posterior Mean (feasibility-weighted).

    Computes the analytic Posterior Mean for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.
    """

    def __init__(
            self,
            model: Model,
            objective: Optional[MCAcquisitionObjective] = None,
            maximize: bool = True,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = objective
        self.posterior_transform = None
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        mean_obj = means[..., 0]
        mean_constraints = means[..., 1:]
        sigma_constraints = sigmas[..., 1:]
        limits = torch.tensor([0] * (means.shape[-1] - 1))
        z = (limits - mean_constraints) / sigma_constraints
        probability_feasibility = torch.distributions.Normal(0, 1).cdf(z).prod(dim=-1)
        constrained_posterior_mean = mean_obj * probability_feasibility
        return constrained_posterior_mean.squeeze(dim=-1)


class CustomGaussianLikelihood(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_constraint("noise_constraint", GreaterThan(1e-4))


class ConstrainedGPModelWrapper():
    def __init__(self, num_constraints):
        self.model_f = None
        self.model = None
        self.num_constraints = num_constraints
        self.train_var_noise = torch.tensor(1e-4, device=device, dtype=dtype)

    def fit(self, X, Y):
        assert Y.shape[1] == self.num_constraints + 1, "missmatch constraint number"
        assert Y.shape[0] == X.shape[0], "missmatch number of evaluations"


        self.model_f = SingleTaskGP(train_X=X,
                                    train_Y=Y[:, 0].reshape(-1, 1),
                                    train_Yvar=self.train_var_noise.expand_as(Y[:, 0].reshape(-1, 1)),
                                    outcome_transform=Standardize(m=1))

        list_of_models = [self.model_f]
        for c in range(1, self.num_constraints + 1):
            list_of_models.append(SingleTaskGP(train_X=X,
                                               train_Y=Y[:, c].reshape(-1, 1),
                                               train_Yvar=self.train_var_noise.expand_as(Y[:, 0].reshape(-1, 1))))

            self.model = ModelListGP(*list_of_models)
            return self.model

    def optimize(self):
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model

class ConstrainedDeoupledGPModelWrapper():
    def __init__(self, num_constraints):
        self.model_f = None
        self.model = None
        self.num_constraints = num_constraints
        self.train_var_noise = torch.tensor(1e-4, device=device, dtype=dtype)

    def fit(self, X, Y):
        assert Y.shape[1] == self.num_constraints + 1, "missmatch constraint number"
        assert Y.shape[0] == X.shape[0], "missmatch number of evaluations"


        self.model_f = SingleTaskGP(train_X=X,
                                    train_Y=Y[:, 0].reshape(-1, 1),
                                    train_Yvar=self.train_var_noise.expand_as(Y[:, 0].reshape(-1, 1)),
                                    outcome_transform=Standardize(m=1))

        list_of_models = [self.model_f]
        for c in range(1, self.num_constraints + 1):
            list_of_models.append(SingleTaskGP(train_X=X,
                                               train_Y=Y[:, c].reshape(-1, 1),
                                               train_Yvar=self.train_var_noise.expand_as(Y[:, 0].reshape(-1, 1))))

            self.model = ModelListGP(*list_of_models)
            return self.model

    def optimize(self):
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model