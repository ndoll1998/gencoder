import multiprocessing as mp
from typing import Any, Optional

import math
import torch
import transformers
from enum import Enum
from dataclasses import dataclass
from transformers.data.data_collator import (
    DataCollatorMixin,
    pad_without_fast_tokenizer_warning,
)

class MaskingStrategy(Enum):
    ALL = "all"
    BERNOULLI = "bernoulli"
    UNIFORM = "uniform"

@dataclass
class DataCollatorForMaskedGenerativeLM(DataCollatorMixin):

    tokenizer: transformers.PreTrainedTokenizerBase
    masking_strategy: MaskingStrategy
    return_tensors: str = "pt"

    def __post_init__(self):
        self.safe_mask_prob = mp.Value("d", 0.15)

    def set_mask_prob(self, prob: float):
        self.safe_mask_prob.value = prob

    def get_mask_prob(self) -> float:
        return self.safe_mask_prob.value

    def torch_call(self, examples: list[dict[str, Any]]):
        
        examples = {
            "input_ids": [e["input_ids"] + e["target_ids"] for e in examples],
            "target_offset": [len(e["input_ids"]) for e in examples],
            "target_length": [len(e["target_ids"]) for e in examples]
        }

        # pad joined sequence to 512
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=512
        )

        n, m = batch["input_ids"].size()
        
        offset = batch.pop("target_offset")
        length = batch.pop("target_length")

        if self.masking_strategy == MaskingStrategy.BERNOULLI:
            batch["input_ids"], batch["labels"] = self.bernoulli(
                batch["input_ids"], offset, length
            )

        elif self.masking_strategy == MaskingStrategy.UNIFORM:
            batch["input_ids"], batch["labels"] = self.uniform(
                batch["input_ids"], offset, length
            )
        
        elif self.masking_strategy == MaskingStrategy.ALL:
            batch["input_ids"], batch["labels"] = self.mask_all(
                batch["input_ids"], offset, length
            )

        # no padding tokens
        pad_mask = batch["input_ids"] == self.tokenizer.pad_token_id
        batch["input_ids"][pad_mask] = self.tokenizer.mask_token_id
        # fill attention mask with all ones
        batch["attention_mask"].fill_(1)

        return batch

    def bernoulli(self, inputs, offset, length) -> tuple[Any, Any]:

        offset = offset.unsqueeze(1)
        length = length.unsqueeze(1)

        n, m = inputs.size()
        base_index = torch.arange(m).unsqueeze(0).expand(n, -1) - offset
        completion_mask = (0 <= base_index) & (base_index < length)

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.safe_mask_prob.value)

        probability_matrix.masked_fill_(~completion_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return inputs, labels

    def uniform(self, inputs, offset, length):
        
        labels = inputs.clone()
        
        num_mask_tokens = torch.rand(length.size()) * length * self.safe_mask_prob.value
        num_mask_tokens = num_mask_tokens.ceil().to(torch.int)
        # at least one token must be masked
        num_mask_tokens[num_mask_tokens < 1] = 1
        max_num_mask_tokens = num_mask_tokens.max()

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for i, (o, l, n) in enumerate(zip(offset, length, num_mask_tokens)):
            # sample the indices to mask
            cur_mask_indices = o + torch.randperm(l)[:n]
            masked_indices[i, cur_mask_indices] = True

        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return inputs, labels

    def mask_all(self, inputs, offset, length):
        
        labels = inputs.clone()
        
        offset = offset.unsqueeze(1)
        length = length.unsqueeze(1)
        
        n, m = inputs.size()
        base_index = torch.arange(m).unsqueeze(0).expand(n, -1) - offset
        masked_indices = (0 <= base_index) & (base_index < length)
        
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return inputs, labels
        


@dataclass
class CosineMaskProbScheduler(transformers.TrainerCallback):

    data_collator: DataCollatorForMaskedGenerativeLM
    init_mask_prob: float
    final_mask_prob: float

    def on_train_begin(self, args, state, control, **kwargs):
        self.data_collator.set_mask_prob(self.init_mask_prob)

    def on_step_begin(self, args, state, control, **kwargs):
        # map progress to consine function
        progress = state.global_step / args.max_steps
        weight = max(0.0, 0.5 * (1.0 - math.cos(math.pi * progress)))
        # compute and update mask prob
        prob = self.init_mask_prob + (self.final_mask_prob - self.init_mask_prob) * weight
        self.data_collator.set_mask_prob(prob)
