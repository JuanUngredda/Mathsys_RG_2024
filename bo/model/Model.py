import torch
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.models.transforms import Standardize
from gpytorch import ExactMarginalLogLikelihood

# constants
device = torch.device("cpu")
dtype = torch.float64


class GPModelWrapper():
    def __init__(self):
        self.train_yvar = torch.tensor(1e-6 ** 2, device=device, dtype=dtype)

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
