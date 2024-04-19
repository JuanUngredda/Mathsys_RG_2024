from typing import Optional
import json
import os
import time

import torch
from botorch.acquisition import MCAcquisitionObjective
from botorch.optim import optimize_acqf
from botorch.test_functions.base import BaseTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor

from bo.acquisition_functions.acquisition_functions import acquisition_function_factory, AcquisitionFunctionType
from bo.model.Model import GPModelWrapper, ConstrainedPosteriorMean

# constants
device = torch.device("cpu")
dtype = torch.float64


class OptimizationLoop:

    def __init__(self, black_box_func: BaseTestProblem,
                 model: GPModelWrapper,
                 objective: Optional[MCAcquisitionObjective],
                 ei_type: AcquisitionFunctionType,
                 seed: int,
                 budget: int,
                 performance_type: str,
                 bounds: Tensor):
        torch.random.manual_seed(seed)
        self.objective = objective
        self.bounds = bounds
        self.black_box_func = black_box_func
        self.dim_x = self.black_box_func.dim
        self.seed = seed
        self.model = model
        self.budget = budget
        self.performance_type = performance_type
        self.acquisition_function_type = ei_type

    def save_parameters(self):

        self.unix_time = round(time.time())
        print(f'Current time: {self.unix_time}')

        # Create folder for saving data.
        folder_path = os.getcwd() + '/data/'
        self.folder = os.path.join(folder_path, 'sim-' + str(self.unix_time))
        os.mkdir(self.folder)

        parameters = {
            'objective': self.objective._get_name(),
            'black_box': self.black_box_func._get_name(),
            'acqf': self.acquisition_function_type.name,
            'seed': self.seed,
            'budget': self.budget,
            'p_type': self.performance_type,
        }
        with open(f'{self.folder}/parameters.json', 'w') as fp:
            json.dump(parameters, fp)
    
    def run(self):

        self.save_parameters()

        best_observed_all_sampled = []

        train_x, train_y = self.generate_initial_data(n=6)

        model = self.update_model(train_x, train_y)
        for iteration in range(self.budget):
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)
            acquisition_function = acquisition_function_factory(model=model,
                                                                type=self.acquisition_function_type,
                                                                objective=self.objective,
                                                                best_value=best_observed_value)

            new_x = self.compute_next_sample(acquisition_function=acquisition_function) # Coupled
            new_y = self.evaluate_black_box_func(unnormalize(new_x, bounds=self.bounds))

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])
            model = self.update_model(X=train_x, y=train_y)

            print(
                f"\nBatch {iteration:>2}: best_value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location) + " current sample decision x: " + str(new_x),
                end="",
            )
            with open(f'{self.folder}/results.dat', 'a') as results_file:
                # TODO: Only works for dimension 2 atm.
                results_file.write(f'{iteration:>2}, {best_observed_value:>4.5f}, {best_observed_location[0][0]}, {best_observed_location[0][1]}, {new_x[0][0]}, {new_x[0][1]}\n')
            
            #for i in range(model.num_constraints + 1):
            #    with open(f'{self.folder}/gp_{i}.dat', 'a') as model_file:
            #        model_file.write()

    def evaluate_black_box_func(self, X):
        return self.black_box_func.evaluate_black_box(X)

    def generate_initial_data(self, n: int):
        # generate training data
        train_x_list = []
        train_y_list = []
        for i in range(self.model.num_outputs) :
            train_x = torch.rand(n, self.dim_x, device=device, dtype=dtype)
            train_x_list += [train_x]
            train_y_list += [self.evaluate_black_box_func(train_x, i)]

        return train_x_list, train_y_list

    def update_model(self, X, y):
        self.model.fit(X, y)
        optimized_model = self.model.optimize()
        return optimized_model

    def best_observed(self, best_value_computation_type, train_x, train_y, model, bounds):
        if best_value_computation_type == "sampled":
            return self.compute_best_sampled_value(train_x, train_y)
        elif best_value_computation_type == "model":
            return self.compute_best_posterior_mean(model, bounds)

    def compute_best_sampled_value(self, train_x, train_y):
        return train_x[torch.argmax(train_y)], torch.max(train_y)

    def compute_best_posterior_mean(self, model, bounds):
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=2048,
        )
        return argmax_mean, max_mean

    def compute_next_sample(self, acquisition_function):
        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,  # used for intialization heuristic
            options={"maxiter": 200},
        )
        # observe new values
        new_x = candidates.detach()
        return new_x
