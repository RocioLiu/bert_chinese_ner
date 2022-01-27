import torch
import torch.nn as nn
from typing import List, Optional

class CRF(nn.Module):
    """
    Conditional random field
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions: torch.nn.Parameter. Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions : torch.nn.Parameter. End transition score tensor of size
            ``(num_tags,)``.
        transitions: torch.nn.Parameter. Transition score tensor of size
            ``(num_tags, num_tags)``.
    """

    def __init__(self, num_tags:ints, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()


    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"


    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum'
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emission: torch.Tensor. Emission score tensor of size (seq_length, batch_size, num_tags) if batch_first is False.
            tags: torch.LongTensor. Sequence of tags tensor of size (seq_length, batch_size) if batch_first is False.
            mask: torch.ByteTensor. Mask tensor of size (seq_length, batch_size) if batch_first is False.
            reduction: Specifies  the reduction to apply to the output: none|sum|mean|token_mean.
                `none`: no reduction will be applied. `sum`: the output will be summed over batches.
                `mean`: the output will be averaged over batches. `token_mean`: the output will be averaged over tokens.

        Returns: torch.Tensor. The log likelihood. This will have size (batch_size,) if reduction is `none`,
            `()` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte() # self.byte() is equivalent to self.to(torch.uint8)

        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            # exchange dim 0 with dim 1, it's the same as emissions.transpose(1, 0)
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)



        def _validate(
                self,
                emissions: torch.Tensor,
                tags: Optional[torch.LongTensor] = None,
                mask: Optional[torch.ByteTensor] = None
        ) -> None:
            if emissions.dim() != 3:
                raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")
            if emissions.size(2) != self.num_tag:
                raise ValueError(
                    f"expected last dimension of emissions is {self.num_tag}, "
                    f"got {emissions.size(2)}"
                )

            if tags is not None:
                if emissions.shape[:2] != tags.shape:
                    raise ValueError(
                        "the first two dimensions of emissions and tags must match, "
                        f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                    )

            if mask is not None:
                if emissions.shape[:2] != mask.shape:
                    raise ValueError(
                        "the first two dimensions of emissions and mask must match, "
                        f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                    )
                # .all() Tests if all elements evaluate to True.
                # take batch_size number first element of a seq (first timestep)
                no_empty_seq = not self.batch_first and mask[0].all()
                no_empty_seq_bf = self.batch_first and mask[:, 0].all()
                if not no_empty_seq and not no_empty_seq_bf:
                    raise ValueError("mask of the first timestep must all be on")



