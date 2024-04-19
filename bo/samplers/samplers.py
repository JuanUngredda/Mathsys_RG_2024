from typing import Optional

import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.sampling.normal import NormalMCSampler
from torch import Tensor
from torch.quasirandom import SobolEngine
from itertools import product


class cKGSampler(NormalMCSampler):
    def forward(self, posterior: Posterior) -> Tensor:
        return super().forward(posterior)

    def _construct_base_samples(self, posterior: Posterior) -> None:
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            base_collapsed_shape = target_shape[len(self.sample_shape):]
            output_dim = base_collapsed_shape.numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = self.draw_quantiles(
                d=output_dim,
                n=self.sample_shape.numel(),
                device=posterior.device,
                dtype=posterior.dtype,
            )
            base_samples = base_samples.view(target_shape)
            self.register_buffer("base_samples", base_samples)
        self.to(device=posterior.device, dtype=posterior.dtype)

    def draw_quantiles(self, d, n, device, dtype):
        base_samples_single_dimension = self.construct_z_vals(nz=n, device=device)
        combinations = list(product(*[base_samples_single_dimension for _ in range(d)]))
        return torch.tensor(combinations, dtype=dtype, device=device)

    def construct_z_vals(self, nz: int, device: Optional[torch.device] = None) -> Tensor:
        """make nz random z """
        quantiles_z = (torch.arange(nz) + 0.5) * (1 / nz)
        normal = torch.distributions.Normal(0, 1)
        z_vals = normal.icdf(quantiles_z)
        return z_vals.to(device=device)

    def _update_base_samples(self, posterior: Posterior, base_sampler: NormalMCSampler) -> None:
        super()._update_base_samples(posterior, base_sampler)

class quantileSampler(NormalMCSampler) : 
    def __init__(self, sample_shape: torch.Size, seed: int | None = None, **kwargs: torch.Any) -> None:
        super().__init__(sample_shape, seed, **kwargs)
    
    def _construct_base_samples(self, posterior: Posterior) -> None:
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            base_collapsed_shape = target_shape[len(self.sample_shape):]
            output_dim = base_collapsed_shape.numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = self.draw_quantiles(
                d=output_dim,
                n=self.sample_shape.numel(),
                device=posterior.device,
                dtype=posterior.dtype,
            )
            base_samples = base_samples.view(target_shape)
            self.register_buffer("base_samples", base_samples)
        self.to(device=posterior.device, dtype=posterior.dtype)

    def draw_quantiles(self, d, n, device, dtype):
        return self.construct_z_vals(nz=n, device=device)
    
    def construct_z_vals(self, nz: int, device: Optional[torch.device] = None) -> Tensor:
        """make nz random z """
        quantiles_z = (torch.arange(nz) + 0.5) * (1 / nz)
        normal = torch.distributions.Normal(0, 1)
        z_vals = normal.icdf(quantiles_z)
        return z_vals.to(device=device)