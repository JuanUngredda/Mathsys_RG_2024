import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, DecoupledAcquisitionFunction
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler, ListSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.acquisition import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from typing import Optional

from bo.acquisition_functions.acquisition_functions import MathsysExpectedImprovement, DecoupledConstrainedKnowledgeGradient
from bo.constrained_functions.synthetic_problems import testing_function
from bo.samplers.samplers import quantileSampler


class TestMathsysExpectedImprovement(BotorchTestCase):
    def test_forward_evaluation(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5]], device=self.device, dtype=dtype)
            variance = torch.ones(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            ei_expected = torch.tensor([0.1978], dtype=dtype)
            X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
            module = MathsysExpectedImprovement(model=mm, best_f=0.0)
            ei_actual = module(X)

            self.assertAllClose(ei_actual, ei_expected, atol=1e-4)
            self.assertEqual(ei_actual.shape, torch.Size([1]))

    def test_forward_shape(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5], [0.7]], device=self.device, dtype=dtype)
            variance = torch.ones(2, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            X = torch.empty(2, 1, 1, device=self.device, dtype=dtype)  # dummy
            # module =? initialize your acquisition function
            # ei_actual = module(X)

            # self.assertTrue(ei_actual.shape == torch.Size([2]))


class TestMathsysMCExpectedImprovement(BotorchTestCase):
    def test_forward_shape(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # the event shape is `b x q x t` = 2 x 1 x 1
            samples = torch.zeros(2, 1, 1, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 2 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(2, 1, 1, **tkwargs)

            # basic test
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
            # acqf = ?
            # ei_actual = acqf(X)

            # self.assertTrue(ei_actual.shape == torch.Size([2]))


class NumericalTest(BotorchTestCase):
    def test_acquisition_functions_are_equivalent_single_run(self):
        d = 2
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(5, d, device=self.device, dtype=dtype)
        train_Y = torch.rand(5, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)

        X = torch.rand(10, 1, d, device=self.device, dtype=dtype)
        ei = ExpectedImprovement(model=model, best_f=0)
        ei_val = ei(X)

        sampler = IIDNormalSampler(sample_shape=torch.Size([1000000]))
        mc_ei = qExpectedImprovement(model=model, best_f=0, sampler=sampler)
        mc_ei_val = mc_ei(X)

        self.assertAllClose(ei_val, mc_ei_val, atol=1e-3)


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
        return Z[..., 0]

class TestDecoupledKG(BotorchTestCase):

    def test_constraints(self):

        dtype = torch.double
        d = 1
        num_points_objective = 50
        num_points_constraint = 5
        expected_decision = 1 # Objective

        torch.manual_seed(0)
        train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
        train_X_constraint = torch.rand(num_points_constraint, d, device=self.device, dtype=dtype)
        func = testing_function()
        train_Y_objective = func.evaluate_true(train_X_objective)
        train_Y_constraint = func.evaluate_slack_true(train_X_constraint)
        NOISE = torch.tensor(1e-9, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X_objective, train_Y_objective, train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint = SingleTaskGP(train_X_constraint, train_Y_constraint, train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))

        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        sampler = quantileSampler(sample_shape=torch.Size([5]))
        sampler_list = ListSampler(*[sampler, sampler])

        kg_values = torch.zeros(2, dtype=dtype)
        for i in range(2):

            acqf = DecoupledConstrainedKnowledgeGradient(model, sampler = sampler_list, num_fantasies=5, 
                                                         objective=ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable]))
            rd = torch.rand(6, 1, d, dtype = dtype)
            acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space
        
            bounds = torch.tensor([[0.0]*d,[1.0]*d], dtype=torch.double)
            candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 15, options={'maxiter': 200})
            kg_values[i] = candidates_values
            print(candidates.shape, candidates_values.shape)
        self.assertEqual(expected_decision, torch.argmax(kg_values))
