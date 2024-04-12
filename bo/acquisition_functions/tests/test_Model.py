from unittest import TestCase

import matplotlib.pyplot as plt
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.acquisition_functions import DecoupledConstrainedKnowledgeGradient
from bo.constrained_functions.synthetic_problems import ConstrainedBranin
from bo.model.Model import ConstrainedGPModelWrapper, ConstrainedPosteriorMean

device = torch.device("cpu")
dtype = torch.double


class TestMathsysExpectedImprovement(BotorchTestCase):

    def test_Branin(self):
        problem = ConstrainedBranin()

        def black_box_evaluation(X):
            y = problem(X).reshape(-1, 1)
            c1 = problem.evaluate_slack_true(X)
            return torch.concat([y, c1], dim=1)

        d = 2
        n_points = 40
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        train_X = torch.concat([train_X, torch.tensor([[0.0, 0.0]])])
        test_X = torch.rand(1000, d, device=self.device, dtype=dtype)
        eval = black_box_evaluation(train_X)

        unfeas_x = train_X[eval[:, 1] >= 0]
        plt.scatter(train_X[:, 0], train_X[:, 1], c=eval[:, 0])
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.show()

        model = ConstrainedGPModelWrapper(num_constraints=1)
        model.fit(train_X, eval)
        optimized_model = model.optimize()

        posterior_distribution = optimized_model.posterior(test_X)
        mean, var = posterior_distribution.mean, posterior_distribution.variance

        unfeas_x = test_X[mean[:, 1] >= 0]

        # plt.scatter(black_box_evaluation(test_X)[:, 1], mean[:, 1].detach())
        # plt.show()
        # plt.scatter(black_box_evaluation(test_X)[:, 0], mean[:, 0].detach())
        # plt.show()
        plt.scatter(test_X[:, 0], test_X[:, 1], c=mean[:, 0].detach())
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.show()


class TestPosteriorConstrainedMean(BotorchTestCase):

    def test_forward(self):
        problem = ConstrainedBranin()

        def black_box_evaluation(X):
            y = problem(X).reshape(-1, 1)
            c1 = problem.evaluate_slack_true(X)
            return torch.concat([y, c1], dim=1)

        d = 2
        n_points = 40
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        train_X = torch.concat([train_X, torch.tensor([[0.0, 0.0]])])
        test_X = torch.rand(1000, d, device=self.device, dtype=dtype)
        eval = black_box_evaluation(train_X)

        model = ConstrainedGPModelWrapper(num_constraints=1)
        model.fit(train_X, eval)
        optimized_model = model.optimize()

        constrained_posterior = ConstrainedPosteriorMean(model=optimized_model, maximize=True)
        penalised_posterior_values = constrained_posterior.forward(test_X[:, None, :])

        plt.scatter(test_X[:, 0], test_X[:, 1], c=penalised_posterior_values.detach())
        plt.show()


class TestConstrainedGPModelWrapper(TestCase):
    def test_fit(self):
        d = 1
        n_points_objective = 10
        n_points_constraints = 6
        torch.manual_seed(0)
        train_Xf = torch.rand(n_points_objective, d, device=device, dtype=dtype)
        train_Xc = torch.rand(n_points_constraints, d, device=device, dtype=dtype)
        problem = ConstrainedBranin()
        train_f_vals = problem.evaluate_true(train_Xf)
        train_c_vals = problem.evaluate_slack_true(train_Xc)

        train_var_noise = torch.tensor(1e-6, device=device, dtype=dtype)
        model_f = SingleTaskGP(train_X=train_Xf,
                               train_Y=train_f_vals.reshape(-1, 1),
                               train_Yvar=train_var_noise.expand_as(train_f_vals.reshape(-1, 1)),
                               outcome_transform=Standardize(m=1))
        model_c = SingleTaskGP(train_X=train_Xc,
                               train_Y=train_c_vals.reshape(-1, 1),
                               train_Yvar=train_var_noise.expand_as(train_c_vals.reshape(-1, 1)))

        model = ModelListGP(model_f, model_c)

        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        kg = DecoupledConstrainedKnowledgeGradient(model=model, num_fantasies=5,
                                                   current_value=torch.Tensor([0.0]))

