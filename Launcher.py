from typing import Optional

import torch
from botorch.acquisition import ConstrainedMCObjective

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loop import OptimizationLoop
from bo.model.Model import ConstrainedGPModelWrapper, ConstrainedDeoupledGPModelWrapper
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedBranin

device = torch.device("cpu")
dtype = torch.double


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable_wrapper(constraint_idx):
    def constraint_callable(Z):
        return Z[..., constraint_idx]

    return constraint_callable


if __name__ == "__main__":
    # Note: the launcher assumes that all inequalities are less than and the limit of the constraint is zero.
    # Transform accordingly in the problem.
    # TODO: save the information bayesian optimization information.
    # TODO: change the penalty M in the acquisition function.
    # TODO: select and code up the test functions to use for the poster/report.
    #
    black_box_function = ConstrainedBranin(noise_std=1e-6, negate=True)
    num_constraints = 1
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    loop = OptimizationLoop(black_box_func=black_box_function,
                            objective=constrained_obj,
                            ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                            performance_type="model",
                            model=model,
                            seed=0,
                            budget=5)
    loop.run()
