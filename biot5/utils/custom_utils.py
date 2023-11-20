from typing import Dict, List
import numpy as np
from transformers import BatchEncoding
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch
import random


@dataclass
class DataCollatorForUnimptT5:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # flatten list of dicts in tuple to list of dicts
        if isinstance(examples[0], tuple):
            examples = [sample_dict for sample_tuple in examples for sample_dict in sample_tuple]

        examples_mlm = []
        examples_src_tgt = []
        for example_i in examples:
            if "labels" in example_i:
                examples_src_tgt.append(example_i)
            else:
                examples_mlm.append(example_i)
        
        batch = BatchEncoding(
            {
                k: np.array([examples_mlm[i][k] for i in range(len(examples_mlm))])
                for k, v in examples_mlm[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch_src_tgt = BatchEncoding(
            {
                k: np.array([examples_src_tgt[i][k] for i in range(len(examples_src_tgt))])
                for k, v in examples_src_tgt[0].items()
            }
        )
        batch_src_tgt['labels'][batch_src_tgt['labels'] == self.pad_token_id] = -100
        # pad batch['labels'] to the same the batch['input_ids']
        batch['labels'] = np.concatenate((batch['labels'], np.full((batch_size, self.input_length - self.target_length), -100)), axis=1)

        batch = {k: np.concatenate((batch[k], batch_src_tgt[k]), axis=0) for k in batch}

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        # sentinel_ids = np.where(
        #     sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        # )
        # For additional molecule and protein tokens
        sentinel_ids = np.where(
            sentinel_ids != 0, (self.tokenizer.vocab_size - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def tokenize_function_seq_desc(examples, tokenizer, max_length):
    tokenizer_seq_out = tokenizer(
        text=examples["seq"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    tokenizer_desc_out = tokenizer(
        text=examples["desc"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    if random.random() < 0.5:
        input_ids = tokenizer_seq_out["input_ids"]
        label_ids = tokenizer_desc_out["input_ids"]
    else:
        input_ids = tokenizer_desc_out["input_ids"]
        label_ids = tokenizer_seq_out["input_ids"]

    result = {"input_ids": input_ids, "labels": label_ids}

    return result
