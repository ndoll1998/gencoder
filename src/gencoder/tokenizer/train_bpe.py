import datasets
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers
)
from transformers import PreTrainedTokenizerFast
from tempfile import TemporaryFile
from typing import Iterator

def batch_iter(ds: datasets.IterableDataset, text_key: str, batch_size: int) -> Iterator[str]:
    for batch in ds.iter(batch_size):
        yield batch[text_key]

def train_bpe_tokenizer(
    ds: datasets.IterableDataset,
    vocab_size: int
) -> PreTrainedTokenizerFast:
    
    # create tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    # create trainer and train tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<CLS>", "<SEP>", "<MASK>"],
    )
    tokenizer.train_from_iterator(
        batch_iter(ds, "text", 1000),
        trainer=trainer
    )
    
    # convert tokenizer to transformers format and save in output directory
    with TemporaryFile(mode='w+t') as temp_file:
        tokenizer.save_(temp_file)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=temp_file)

    return tokenizer
