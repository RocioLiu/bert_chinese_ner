import torch
from torch import nn
from typing import Optional


class CRF(nn.Module):
    """Conditional random field
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

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
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
            mask: Optional[torch.BoolTensor] = None,
            reduction: str = 'sum'
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions: torch.Tensor. Emission score tensor of size
                (seq_length, batch_size, num_tags) if batch_first is False.
            tags: torch.LongTensor. Sequence of tags tensor of size
                (seq_length, batch_size) if batch_first is False.
            mask: torch.BoolTensor. Mask tensor of size
                (seq_length, batch_size) if batch_first is False.
            reduction: Specifies  the reduction to apply to the output:
                none | sum | mean | token_mean.
                `none`: no reduction will be applied.
                `sum`: the output will be summed over batches.
                `mean`: the output will be averaged over batches.
                `token_mean`: the output will be averaged over tokens.

        Returns: torch.Tensor. The log likelihood. This will have size
            (batch_size,) if reduction is `none`, `()` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool, device=tags.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            # exchange dim 0 with dim 1, it's the same as emissions.transpose(1, 0)
            # if the input data is batch_first, transforms its shape to
            # (seq_length, batch_size, num_tags)
            emissions = emissions.transpose(0, 1)
            # (seq_length, batch_size)
            tags = tags.transpose(0, 1)
            # (seq_length, batch_size)
            mask = mask.transpose(0, 1)


        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        if reduction == 'token_mean':
            return llh.sum() / mask.float().sum()
        raise ValueError(
            f"reduction should be `none`, `sum`, `mean`, or `token_mean`, but got {reduction}")

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.BoolTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> torch.Tensor:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: torch.Tensor. Emission score tensor of size
                (seq_length, batch_size, num_tags) if batch_first is False.
            mask: torch.BoolTensor. Mask tensor of size
                (seq_length, batch_size) if batch_first is False.
            nbest: int. Number of most probable paths for each sequence
            pad_tag: int. Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`

        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool,
                                      device=emissions.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            # (1, batch_size, seq_length)
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)

        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.BoolTensor] = None
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
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
            # .all() tests if all elements evaluate to True.
            # take batch_size number first element of a seq (first timestep)
            no_empty_seq = mask[0, :].all() if (self.batch_first == False) else mask[:, 0].all()
            if not no_empty_seq:
                raise ValueError("mask of the first timestep must all be on")
            # no_empty_seq = not self.batch_first and mask[0].all().item()
            # print(f"no_empty_seq: {no_empty_seq}")
            # no_empty_seq_bf = self.batch_first and mask[:, 0].all().item()
            # print(f"no_empty_seq_bf: {no_empty_seq_bf}")
            # if not no_empty_seq and not no_empty_seq_bf:
            #     raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: torch.BoolTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,) the first element in a seq for each example of a batch
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # we only want to get the active score (i.e. mask[i] == 1)
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (seq_len, batch_size) -> (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # last tag of each example in a batch
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.BoolTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample (a batch), entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.BoolTensor,
                        pad_tag: Optional[int] = None) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags) if batch_first == False
        # mask: (seq_length, batch_size) if batch_first == False
        # return (batch_size, seq_length) if batch_first == False

        if pad_tag is None:
            pad_tag = 0

        device = emissions.device

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags) if batch_first == False
        # (seq_len, num_tags) if batch_first == True
        score = self.start_transitions + emissions[0]

        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # score: a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history_idx: it saves where the best tags candidate transitioned from; this
        # is used when we trace back the best tag sequence
        # oor_idx: it saves the best tags candidate transitioned from at the positions
        # where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # next_score.shape: (batch_size, num_tags)
            # indices is the tag at last timestep which maximize the score at this timestep
            # indices.shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].bool().unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].bool().unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # Now, compute the best path for each sample
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        ## insert the best tag at each sequence end, which we didn't save where its
        ## best tag candidate transitioned from. (last position with mask == 1)

        # we should exchange the axis 0 and axis 1 of history_idx to let its shape
        # match the shape of reshaped seq_ends and end_tag (batch_size first)
        # shape: (batch_size, seq_length, num_tags)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        # The most probable tag at each given word
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            # the best tag at that index in the sequence
            best_tags_arr[idx] = best_tags.view(batch_size)

        # (batch_size, seq_len)
        return torch.where(mask.bool(), best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.BoolTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                # (batch_size, num_tags, nbest)
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                # (batch_size, nbest, num_tags)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].bool().unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].bool().unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)

        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = torch.div(best_tags, nbest, rounding_mode='floor')

        return torch.where(mask.bool().unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)

