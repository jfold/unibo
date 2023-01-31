from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import batched_multinomial
from botorch.utils.transforms import standardize
from torch import Tensor
from torch.nn import Module
import numpy as np


class SamplingStrategy(Module, ABC):
    r"""
    Abstract base class for sampling-based generation strategies.

    :meta private:
    """

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover

class MinPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MinPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        surrogate_type: str = "GP"
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        self.surrogate_type = surrogate_type
        if objective is None:
            objective = IdentityMCObjective()
        elif not isinstance(objective, MCAcquisitionObjective):
            # TODO: Clean up once ScalarizedObjective is removed.
            if posterior_transform is not None:
                raise RuntimeError(
                    "A ScalarizedObjective (DEPRECATED) and a posterior transform "
                    "are not supported at the same time. Use only a posterior "
                    "transform instead."
                )
            else:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=objective.weights, offset=objective.offset
                )
                objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        #ThompsonSampling does not work with other surrogates (due to base class), due to the shape of X. GP can handle 1xNxD, but other models must have Bx1xD. We must reshape X for the posterior, and then reshape the posterior sample before performing maximize samples.
        #Not so elegant but works...
        if self.surrogate_type == "GP":
            posterior = self.model.posterior(
                X,
                observation_noise=observation_noise,
                posterior_transform=self.posterior_transform,
            )
            # num_samples x batch_shape x N x m
            samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
            return self.minimize_samples(X, samples, num_samples)
        elif self.surrogate_type == "BNN":
            X = X.permute(1, 0, 2)
            posterior = self.model.posterior(
                X,
                observation_noise=observation_noise,
                posterior_transform=self.posterior_transform,
            )
            X = X.permute(1, 0, 2)
            # num_samples x batch_shape x N x m
            samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
            samples = samples.unsqueeze(1)
            return self.minimize_samples(X, samples, num_samples)
        elif self.surrogate_type == "DE":
            sampled_MLP_idx = np.random.choice(np.arange(self.model.n_networks))
            sampled_MLP = self.model.models[sampled_MLP_idx]
            X = X.squeeze(0)
            preds = sampled_MLP(X.float())
            return X[np.argmin(preds.detach().numpy())]
        elif self.surrogate_type == "RF":
            sampled_tree_idx = np.random.choice(np.arange(len(self.model.model.estimators_)))
            sampled_tree = self.model.model.estimators_[sampled_tree_idx]
            X = X.squeeze(0)
            preds = sampled_tree.predict(X.detach().numpy())
            return X[np.argmin(preds)]





    def minimize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmin(obj, dim=-1)
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)