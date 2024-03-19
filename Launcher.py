import warnings

import torch
from botorch.test_functions import Branin

from bo.acquisition_functions.acquisition_functions import EIType
from bo.bo_loop import OptimizationLoop
from bo.model.Model import GPModelWrapper

device = torch.device("cpu")
dtype = torch.double

if __name__ == "__main__":
    bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], device=device, dtype=dtype)
    black_box_function = Branin(noise_std=1e-6, negate=True)
    model = GPModelWrapper()
    loop = OptimizationLoop(black_box_func=black_box_function,
                            ei_type=EIType.BOTORCH_EXPECTED_IMPROVEMENT,
                            bounds=bounds,
                            model=model,
                            seed=0,
                            budget=30)
    loop.run()
