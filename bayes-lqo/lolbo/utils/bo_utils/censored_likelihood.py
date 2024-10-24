#!/usr/bin/env python3

from typing import Any, Union

import torch
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from linear_operator.operators import LinearOperator
from torch import Tensor
from torch.distributions import Normal


class CensoredGaussianLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Censored likelihood used for Censored GP
    """

    def __init__(self, censored_obs_is_max=False):
        super().__init__()
        self.censored_obs_is_max = censored_obs_is_max
        #   If True, the censored observaation is the max possible value (it could be lower)
        #   If False, the censored observaation is the MIN possible value (it could be higher) (version from paper)
        self.noise_covar = HomoskedasticNoise(noise_prior=None, noise_constraint=None, batch_shape=torch.Size())

    def forward(self, function_samples: Tensor, censored=None, censoring=None, *params: Any, **kwargs: Any) -> Normal:
        if censored is None:
            raise RuntimeError("Censored can't be none, specify true or false")
        if censoring is None:
            raise RuntimeError("Censoring can't be none")

        # function_samples.shape ::  torch.Size([20, 100])

        # Use censoring tensor to seperate out censored vs non censored samples
        censoring = censoring.squeeze()  # torch.Size([100])
        total_n_samples = function_samples.shape[-1]  # 100
        assert total_n_samples > 0
        censored_samples = function_samples[:, censoring == 1]  # torch.Size([20, 54])
        uncensored_samples = function_samples[:, censoring == 0]  # torch.Size([20, 46])
        num_censored = censored_samples.shape[-1]  # 54
        num_uncensored = uncensored_samples.shape[-1]  # 46
        assert (num_censored + num_uncensored) == total_n_samples

        # Return normal dist
        if censored:
            noise = self._shaped_noise_covar(censored_samples.shape, *params, **kwargs).diagonal(
                dim1=-1, dim2=-2
            )  # torch.Size([20, 54])
            dist_censored = base_distributions.Normal(censored_samples, noise.sqrt())  # BEST!
            return dist_censored  # Normal(loc: torch.Size([20, 54]), scale: torch.Size([20, 54]))
        else:
            noise = self._shaped_noise_covar(uncensored_samples.shape, *params, **kwargs).diagonal(
                dim1=-1, dim2=-2
            )  # torch.Size([20, 46])
            dist_uncensored = base_distributions.Normal(uncensored_samples, noise.sqrt())
            return dist_uncensored  # Normal(loc: torch.Size([20, 46]), scale: torch.Size([20, 46]))

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, censoring=None, *args: Any, **kwargs: Any
    ) -> Tensor:
        if censoring is None:
            raise RuntimeError("Censoring can't be none")

        # Use censoring tensor to seperate out censored vs non censored samples
        censoring = censoring.squeeze()
        total_n_obs = observations.shape[-1]
        assert total_n_obs > 0
        censored_obs = observations[censoring == 1]
        uncensored_obs = observations[censoring == 0]
        num_censored = censored_obs.shape[-1]
        num_uncensored = uncensored_obs.shape[-1]
        assert (num_censored + num_uncensored) == total_n_obs

        # For uncensored samples
        if num_uncensored > 0:
            prob_lambda = (
                lambda function_samples: self.forward(
                    function_samples, censored=False, censoring=censoring, *args, **kwargs
                )
                .log_prob(uncensored_obs)
                .exp()
            )
            prob_uncensored = self.quadrature(prob_lambda, function_dist)
            prob_uncensored = prob_uncensored.log()
            if num_censored == 0:
                return prob_uncensored
        # For censored samples: prob = 1 - Normal(f, noise).cdf(observations)
        if num_censored > 0:
            if self.censored_obs_is_max:
                # the censored observaation is the max possible value (it could be lower)
                prob_lambda = lambda function_samples: self.forward(
                    function_samples, censored=True, censoring=censoring, *args, **kwargs
                ).cdf(censored_obs)
            else:
                # the censored observaation is the MIN possible value (it could be higher)
                prob_lambda = lambda function_samples: 1 - self.forward(
                    function_samples, censored=True, censoring=censoring, *args, **kwargs
                ).cdf(censored_obs)
            prob_censored = self.quadrature(prob_lambda, function_dist)
            prob_censored = prob_censored.log()
            if num_uncensored == 0:
                return prob_censored

        # Return all probs
        return torch.cat((prob_uncensored, prob_censored))

    # Directly copied from: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/likelihoods/gaussian_likelihood.py
    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise
