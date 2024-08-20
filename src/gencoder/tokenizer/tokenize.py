import datasets
from datasets import Sequence, Value
from typing import Any

from hyped.data.flow import DataFlow, ops
from hyped.data.flow.processors.tokenizers.transformers import TransformersTokenizer

def tokenize_finetune_ds(
    ds: datasets.IterableDataset,
    tokenizer: str,
    max_length: int,
    input_key: str,
    target_key: str
) -> tuple[datasets.IterableDataset, dict[str, Any]]:

    # create the data flow instance
    flow = DataFlow(ds.features)

    # create the tokenizer
    t = TransformersTokenizer(
        tokenizer=tokenizer,
        add_special_tokens=False
    )
    # apply the tokenizer to the input and targets
    input_ids = t.call(text=flow.src_features[input_key]).input_ids
    target_ids = t.call(text=flow.src_features[target_key]).input_ids
    
    # create constants for special tokens
    CLS = flow.const([t.tokenizer.cls_token_id], feature=Sequence(Value("int32")))
    SEP = flow.const([t.tokenizer.sep_token_id], feature=Sequence(Value("int32")))

    # add special tokens to the input and target sequences
    input_ids = ops.chain(CLS, input_ids, SEP)
    target_ids = ops.chain(target_ids, SEP)

    # apply the flow to the dataset
    ds, stats = flow.apply(
        ds,
        collect=ops.collect(
            {
                "input_ids": input_ids,
                "target_ids": target_ids
            }
        ),
        aggregate=ops.collect(
            {
                "num_prompt_tokens": input_ids.length_().sum_(),
                "avg_prompt_tokens": input_ids.length_().mean_(),
                "num_completion_tokens": target_ids.length_().sum_(),
                "avg_completion_tokens": target_ids.length_().mean_()
            }
        )
    )

    # apply the filter separately
    features = ds.features
    ds = ds.filter(lambda e: len(e["input_ids"]) + len(e["target_ids"]) <= max_length)
    ds.info.features = features

    # return the output dataset and statistics
    return ds, stats
