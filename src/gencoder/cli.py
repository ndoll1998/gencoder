import os
import json
from math import ceil
from glob import glob
from pathlib import Path
from itertools import count
from typing import Optional

import typer
import torch
import datasets
import transformers
import torch.distributed.checkpoint as dist_cp
from torch.nn.attention import SDPBackend, sdpa_kernel
from huggingface_hub import snapshot_download

from .tokenizer.train_bpe import train_bpe_tokenizer
from .tokenizer.tokenize import tokenize_finetune_ds
from .utils.arrow_writer import ArrowDatasetWriter
from .modeling.collator import (
    MaskingStrategy,
    DataCollatorForMaskedGenerativeLM,
    CosineMaskProbScheduler
)

app = typer.Typer()

data_app = typer.Typer()
app.add_typer(data_app, name="data")

tokenizer_app = typer.Typer()
app.add_typer(tokenizer_app, name="tokenizer")

model_app = typer.Typer()
app.add_typer(model_app, name="model")


@data_app.command()
def hf_download(
    repo_id: str = typer.Option(..., "-r", "--repo-id"),
    patterns: list[str] = typer.Option([], "-p", "--pattern"),
    save_dir: Path = typer.Option(..., "-s", "--save-dir"),
    cache_dir: Path = typer.Option(None, "-c", "--cache-dir"),
    force: bool = typer.Option(False, "-f", "--force")
) -> None:
    # create the save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    # download the dataset
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=save_dir.as_posix(),
        cache_dir=cache_dir.as_posix(),
        allow_patterns=patterns
    )


@tokenizer_app.command()
def train_bpe(
    data_files: list[Path],
    data_format: str = typer.Option("json", "-f", "--format"),
    vocab_size: int = typer.Option(..., "-n", "--vocab-size"),
    save_dir: Path = typer.Option(..., "-s", "--save-dir"),
) -> None:

    # check all files
    for f in data_files:
        if not f.is_file():
            raise typer.BadParameter(f"Invalid file path: {input_file}")

    # load dataset
    ds = datasets.load_dataset(
        data_format,
        data_files=list(map(Path.as_posix, data_files)),
        split="train",
        streaming=True
    )
    # train the tokenizer
    tokenizer = train_bpe_tokenizer(
        ds=ds,
        vocab_size=vocab_size
    )
    # save the tokenizer
    tokenizer.save_pretrained(save_dir)


@tokenizer_app.command()
def tokenize(
    data_files: list[Path],
    data_format: str = typer.Option("json", "-f", "--format"),
    tokenizer: str = typer.Option(..., "-t", "--tokenizer"),
    max_length: int = typer.Option(..., "-l", "--max-length"),
    num_proc: int = typer.Option(os.cpu_count(), "-p", "--num-proc"),
    input_key: str = typer.Option("inputs", "--input-key"),
    target_key: str = typer.Option("targets", "--target-key"),
    save_dir: Path = typer.Option(..., "-s", "--save-dir"),
) -> None:
    
    # check all files
    for f in data_files:
        if not f.is_file():
            raise typer.BadParameter(f"Invalid file path: {f}")
    
    # load dataset
    ds = datasets.load_dataset(
        data_format,
        data_files=list(map(Path.as_posix, data_files)),
        split="train",
        streaming=True,
        features=datasets.Features(
            {
                input_key: datasets.Value("string"),
                target_key: datasets.Value("string"),
            }
        )
    )

    tokenized_ds, stats = tokenize_finetune_ds(
        ds,
        tokenizer=tokenizer,
        max_length=max_length,
        input_key=input_key,
        target_key=target_key,
    )

    writer = ArrowDatasetWriter(
        save_dir=save_dir.as_posix(),
        exist_ok=False,
        num_proc=num_proc
    )
    writer.consume(tokenized_ds)

    stats = dict(stats)

    print("Prepared Dataset Meta:")
    print(json.dumps(stats, indent=2))

    with open(save_dir / "meta.json", "w") as f:
        f.write(json.dumps(stats, indent=2))
        

@model_app.command()
def finetune(
    train_data: Path = typer.Option(..., "--train-data"),
    test_data: Optional[Path] = typer.Option(None, "--test-data"),
    model_ckpt: str = typer.Option(..., "--model-ckpt"),
    eval_steps: int = typer.Option(500, "--eval-steps"),
    save_steps: int = typer.Option(5000, "--save-steps"),
    logging_steps: int = typer.Option(50, "--logging-steps"),
    batch_size: int = typer.Option(..., "--batch-size"),
    grad_accumulation: int = typer.Option(1, "--grad-accumulation"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    dropout: float = typer.Option(0.1, "--dropout"),
    warmup_steps: int = typer.Option(500, "--warmup-steps"),
    masking_strategy: MaskingStrategy = typer.Option(MaskingStrategy.BERNOULLI, "--masking-strategy"),
    init_mask_prob: float = typer.Option(1.0, "--init-mask-prob"),
    final_mask_prob: float = typer.Option(1.0, "--final-mask-prob"),
    save_dir: Path = typer.Option(..., "-s", "--save-dir"),
) -> None:
    
    print("(RANK %s) STARTED WORKER" % os.environ.get("RANK", "1"))
    print("(RANK %s) Loading Data" % os.environ.get("RANK", "1"))

    train_ds = datasets.Dataset.load_from_disk(train_data.as_posix(), keep_in_memory=False)
    test_ds = (
        None if test_data is None
        else datasets.Dataset.load_from_disk(test_data.as_posix(), keep_in_memory=False)
    )
    
    global_batch_size = (
        batch_size
        * grad_accumulation
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    max_steps = ceil(len(train_ds) / global_batch_size)

    if os.environ.get("LOCAL_RANK", "1") == "1":
        # log some args
        print("Global Batch Size: ", global_batch_size)
        print("Max Training Steps:", max_steps)
    
    # build training arguments
    training_args = transformers.TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="no" if test_ds is None else "steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=warmup_steps,
        bf16=True,
        local_rank=int(os.environ.get("LOCAL_RANK", "-1")),
        remove_unused_columns=False,
        dataloader_num_workers=4,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        accelerator_config={
            "split_batches": False,
            "dispatch_batches": False
        },
        report_to="wandb",
        include_num_input_tokens_seen=True,
    )
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

        print("(RANK %s) Loading Model" % os.environ.get("RANK", "1"))
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_ckpt,
            attn_implementation="sdpa",
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout
        )

        data_collator = DataCollatorForMaskedGenerativeLM(
            tokenizer=tokenizer,
            masking_strategy=masking_strategy
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
        )

        mask_prob_scheduler = CosineMaskProbScheduler(
            data_collator=data_collator,
            init_mask_prob=init_mask_prob,
            final_mask_prob=final_mask_prob
        )

        trainer.add_callback(mask_prob_scheduler)

        print("(Rank %s) Ready for Training" % os.environ.get("RANK", "1"))
        trainer.train()


@model_app.command()
def convert_ckpt(
    model_type: str = typer.Option(..., "--model-type"),
    model_ckpt: Path = typer.Option(..., "--model-ckpt"),
    save_dir: Path = typer.Option(..., "-s", "--save-dir")
) -> None:

    if not model_ckpt.is_dir():
        raise typer.BadParameter(f"Model Checkpoint not found: {model_ckpt}")
    
    # create the save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # create model with random values
    config = transformers.AutoConfig.from_pretrained(model_type)
    model = transformers.AutoModelForMaskedLM.from_config(config)
    # load state dict from distributed checkpoint
    state_dict = {"model": model.state_dict()}
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(model_ckpt),
        #no_dist=True
    )
    # load back into model and save non-distributed checkpoint
    model.load_state_dict(state_dict["model"])
    model.save_pretrained(save_dir.as_posix())
    # save tokenizer in directory
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    tokenizer.save_pretrained(save_dir.as_posix())


@model_app.command()
@torch.inference_mode()
def inference(
    model_ckpt: Path = typer.Option(..., "--model-ckpt"),
    prompts: list[str] = typer.Option(..., "-p", "--prompt"),
    batch_size: None | int = typer.Option(None, "-b", "--batch-size"),
    no_cuda: bool = typer.Option(True, "--no-cuda")
) -> None:
    
    if not model_ckpt.is_dir():
        raise typer.BadParameter(f"Model Checkpoint not found: {model_ckpt}")

    # load the model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        model_ckpt.as_posix(), attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).cuda()

    # pack prompts in dataset
    ds = datasets.Dataset.from_dict(
        {
            "prompt": prompts,
            "output": [""] * len(prompts)
        }
    )

    # apply the preprocessing pipeline
    tokenized_ds, _ = tokenize_finetune_ds(
        ds,
        tokenizer=model_ckpt.as_posix(),
        max_length=512,
        input_key="prompt",
        target_key="output"
    )
   
    # build data collator and set mask probability to 1
    data_collator = DataCollatorForMaskedGenerativeLM(
        tokenizer=tokenizer,
        masking_strategy=MaskingStrategy.ALL
    )
    # build data loader
    dataloader = torch.utils.data.DataLoader(
        tokenized_ds,
        shuffle=False,
        batch_size=batch_size or len(prompts),
        collate_fn=data_collator
    )

    completions = [
        {"prompt": prompt, "completion": None}
        for prompt in prompts
    ]

    counter = count()
    for batch in dataloader:
        # move input ids to cuda device and compute the number of prompt tokens
        input_ids = batch["input_ids"].cuda()
        lengths = (input_ids != tokenizer.mask_token_id).sum(dim=-1).cpu()

        # apply the model and get the predicted tokens
        out = model.forward(
            batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda()
        )
        completion_ids = out.logits.argmax(dim=-1).cpu()

        # decode all samples in the batch
        for i, j in zip(range(completion_ids.size(0)), counter):
            completion = tokenizer.decode(completion_ids[i, lengths[i]:])
            completions[j]["completion"] = completion

    print(json.dumps(completions, indent=2))


if __name__ == '__main__':
    app()
