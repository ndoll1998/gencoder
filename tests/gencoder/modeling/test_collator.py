import torch
import pytest
from transformers import AutoTokenizer
from gencoder.modeling.collator import (
    MaskingStrategy,
    DataCollatorForMaskedGenerativeLM
)
from math import ceil

@pytest.fixture
def examples():
    return [
        {
            "input_ids": [501, 502, 503, 504],
            "target_ids": [601, 602, 603, 604],
        },
        {
            "input_ids": [501, 502, 503, 504, 505],
            "target_ids": [601, 602, 603, 604, 605],
        },
        {
            "input_ids": [501, 502, 503],
            "target_ids": [601, 602, 603, 604, 605],
        },
        {
            "input_ids": [501, 502, 503, 504, 505],
            "target_ids": [601, 602, 603],
        },
    ]

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def test_masking_strategy_all(tokenizer, examples):

    collator = DataCollatorForMaskedGenerativeLM(
        tokenizer=tokenizer,
        masking_strategy=MaskingStrategy.ALL
    )

    output = collator.torch_call(examples)

    labels = output["labels"]
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]

    assert (attention_mask == 1).all()

    for i, example in enumerate(examples):

        n = len(example["input_ids"])
        m = len(example["target_ids"])

        input_prompt_ids = input_ids[i, :n]
        input_target_ids = input_ids[i, n:n+m]
        input_padding_ids = input_ids[i, n+m:]

        label_prompt_ids = labels[i, :n]
        label_target_ids = labels[i, n:n+m]
        label_padding_ids = labels[i, n+m:]

        assert (input_prompt_ids == torch.LongTensor(example["input_ids"])).all()
        assert (input_target_ids == tokenizer.mask_token_id).all()
        assert (input_padding_ids == tokenizer.mask_token_id).all()
        
        assert (label_prompt_ids == -100).all()
        assert (label_target_ids == torch.LongTensor(example["target_ids"])).all()
        assert (label_padding_ids == -100).all()

@pytest.mark.parametrize("mask_prob", [0.15, 0.5, 0.75, 1.0])
def test_masking_strategy_bernoulli(tokenizer, examples, mask_prob):

    collator = DataCollatorForMaskedGenerativeLM(
        tokenizer=tokenizer,
        masking_strategy=MaskingStrategy.BERNOULLI
    )
    collator.set_mask_prob(mask_prob)

    output = collator.torch_call(examples)

    labels = output["labels"]
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]

    assert (attention_mask == 1).all()

    for i, example in enumerate(examples):

        n = len(example["input_ids"])
        m = len(example["target_ids"])

        input_prompt_ids = input_ids[i, :n]
        input_target_ids = input_ids[i, n:n+m]
        input_padding_ids = input_ids[i, n+m:]

        label_prompt_ids = labels[i, :n]
        label_target_ids = labels[i, n:n+m]
        label_padding_ids = labels[i, n+m:]

        assert (input_prompt_ids == torch.LongTensor(example["input_ids"])).all()
        assert (label_prompt_ids == -100).all()
        
        assert (
            (input_target_ids == tokenizer.mask_token_id)
            | (input_target_ids == torch.LongTensor(example["target_ids"]))
        ).all()
        assert (
            (label_target_ids == -100)
            | (label_target_ids == torch.LongTensor(example["target_ids"]))
        ).all()
        
        assert (input_padding_ids == tokenizer.mask_token_id).all()
        assert (label_padding_ids == -100).all()

        # check percentage of masked tokens
        prob = (input_target_ids == tokenizer.mask_token_id).float().mean()
        assert max(0, mask_prob - 0.5) <= prob.item() <= min(1.0, mask_prob + 0.5)


@pytest.mark.parametrize("mask_prob", [0.15, 0.5, 0.75, 1.0])
def test_masking_strategy_uniform(tokenizer, examples, mask_prob):

    collator = DataCollatorForMaskedGenerativeLM(
        tokenizer=tokenizer,
        masking_strategy=MaskingStrategy.UNIFORM
    )
    collator.set_mask_prob(mask_prob)

    output = collator.torch_call(examples)

    labels = output["labels"]
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]

    assert (attention_mask == 1).all()

    for i, example in enumerate(examples):

        n = len(example["input_ids"])
        m = len(example["target_ids"])

        input_prompt_ids = input_ids[i, :n]
        input_target_ids = input_ids[i, n:n+m]
        input_padding_ids = input_ids[i, n+m:]

        label_prompt_ids = labels[i, :n]
        label_target_ids = labels[i, n:n+m]
        label_padding_ids = labels[i, n+m:]

        assert (input_prompt_ids == torch.LongTensor(example["input_ids"])).all()
        assert (label_prompt_ids == -100).all()
        
        assert (
            (input_target_ids == tokenizer.mask_token_id)
            | (input_target_ids == torch.LongTensor(example["target_ids"]))
        ).all()
        assert (
            (label_target_ids == -100)
            | (label_target_ids == torch.LongTensor(example["target_ids"]))
        ).all()
        
        assert (input_padding_ids == tokenizer.mask_token_id).all()
        assert (label_padding_ids == -100).all()
        
        # check number of masked tokens
        num_masked_tokens = (input_target_ids == tokenizer.mask_token_id).float().sum()
        assert num_masked_tokens.item() <= ceil(mask_prob * len(example["target_ids"]))
