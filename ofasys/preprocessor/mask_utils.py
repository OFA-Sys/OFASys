# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math

import torch


def add_whole_word_mask(
    source,
    p,
    mask_span_distribution,
    poisson_lambda,
    random_ratio,
    mask_idx,
    replace_length,
    tgt_dict_size,
):
    """
    Add whole word masking for input texts, followng BART (Lewis et al., 2019).

    Args:
        source: input text
        p: mask ratio
        mask_span_distribution: mask span distribution.
        random_ratio: the ratio of using random tokens instead of '<mask>'.
        mask_idx: the index of '<mask>'.
        replace_length: replace length.
        tgt_dict_size: the size of vocabulary.
        code_dict_size: the size of code vocabulary.

    Returns:
        source: masked text
    """

    def word_starts(source):
        """
        Decide the start position of the word.

        Args:
            source: input text.

        Returns:
            is_word_start: a tensor to judge word start position.
        """
        is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_insertion_noise(tokens, p):
        """
        Add insertion noise

        Args:
            tokens: input tokens.
            p: mask ratio.
            random_ratio: the ratio of using random takons instead of '<mask>'.
            mask_idx: the index of '<mask>'.
            tgt_dict_size: the size of target vocabulary.
            code_dict_size: the size of code vocabulary.

        Returns:
            result: tokens with noise.
        """
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * random_ratio))
        result[noise_indices[num_random:]] = mask_idx
        result[noise_indices[:num_random]] = torch.randint(low=4, high=tgt_dict_size, size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    if mask_span_distribution == "span-poisson":
        _lambda = poisson_lambda

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        mask_span_distribution = torch.distributions.Categorical(ps)

    is_word_start = word_starts(source)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
    num_inserts = 0
    if num_to_mask == 0:
        return source

    if mask_span_distribution is not None:
        lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return add_insertion_noise(source, num_inserts / source.size(0))

        assert (lengths > 0).all()
    else:
        lengths = torch.ones((num_to_mask,)).long()
    assert is_word_start[-1] == 0
    word_starts = is_word_start.nonzero(as_tuple=False)
    indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

    source_length = source.size(0)
    assert source_length - 1 not in indices
    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end of doc
    if replace_length == 0:
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        source[indices] = mask_idx
        source[indices[mask_random]] = torch.randint(4, tgt_dict_size, size=(mask_random.sum(),))

    if mask_span_distribution is not None:
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = mask_idx
                source[indices[mask_random]] = torch.randint(4, tgt_dict_size, size=(mask_random.sum(),))
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = mask_idx
                source[indices[mask_random]] = torch.randint(4, tgt_dict_size, size=(mask_random.sum(),))

            assert source_length - 1 not in indices

    source = source[to_keep]

    if num_inserts > 0:
        source = add_insertion_noise(source, num_inserts / source.size(0))

    return source
