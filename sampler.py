import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle: RayBundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        device = ray_bundle.origins.device
        bin_size = (self.max_depth - self.min_depth) / self.n_pts_per_ray
        unif_samples = torch.rand(self.n_pts_per_ray, device=device) * bin_size
        z_vals = torch.linspace(self.min_depth, self.max_depth - bin_size, self.n_pts_per_ray, device=device) \
            + unif_samples
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1)

        # TODO (Q1.4): Sample points from z values
        sample_points = ray_bundle.origins.unsqueeze(1) + z_vals \
            * (ray_bundle.directions / ray_bundle.directions.norm(dim=-1, keepdim=True)).unsqueeze(1)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}