import torch
from botorch.optim import optimize_acqf
from botorch.test_functions.base import BaseTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor

from bo.acquisition_functions.acquisition_functions import ExpectedImprovementFactory, EIType
from bo.model.Model import GPModelWrapper

# constants
device = torch.device("cpu")
dtype = torch.float64


class OptimizationLoop:

    def __init__(self, black_box_func: BaseTestProblem, model: GPModelWrapper, ei_type: EIType, seed: int, budget: int,
                 bounds: Tensor):
        torch.random.manual_seed(seed)
        self.bounds = bounds
        self.black_box_func = black_box_func
        self.dim_x = self.black_box_func.dim
        self.seed = seed
        self.model = model
        self.budget = budget
        self.ei_type = ei_type

    def run(self):
        best_observed_all_sampled = []

        train_x, train_y = self.generate_initial_data(n=6)

        model = self.update_model(train_x, train_y)

        best_observed_all_sampled.append(self.best_observed(train_y))
        for iteration in range(self.budget):
            acquisition_function = ExpectedImprovementFactory(model=model,
                                                              ei_type=self.ei_type,
                                                              best_value=self.best_observed(train_y))

            new_x = self.compute_next_sample(acquisition_function=acquisition_function)
            new_y = self.evaluate_black_box_func(unnormalize(new_x, bounds=self.bounds))

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])
            model = self.update_model(X=train_x, y=train_y)

            best_observed_all_sampled.append(self.best_observed(train_y))
            print(
                f"\nBatch {iteration:>2}: best_value (EI) = "
                f"({self.best_observed(train_y):>4.5f}), ",
                end="",
            )

    def evaluate_black_box_func(self, X):
        return self.black_box_func(X).reshape(len(X), 1)

    def generate_initial_data(self, n: int):
        # generate training data
        train_x = torch.rand(n, self.dim_x, device=device, dtype=dtype)
        return train_x, self.evaluate_black_box_func(unnormalize(train_x, self.bounds))

    def update_model(self, X, y):
        self.model.fit(X, y)
        optimized_model = self.model.optimize()
        return optimized_model

    def best_observed(self, train_y):
        return torch.max(train_y)

    def compute_next_sample(self, acquisition_function):
        bounds = torch.tensor([[0.0] * self.dim_x, [1.0] * self.dim_x])
        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"maxiter": 200},
        )
        # observe new values
        new_x = candidates.detach()
        return new_x
