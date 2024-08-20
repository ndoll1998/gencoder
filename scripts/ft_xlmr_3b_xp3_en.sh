export WANDB_PROJECT="gen-bert"
export WANDB_MODE="online"

export WANDB_TAGS="model=xlmr-xl,data=xP3-en,optim=bf16,model=fp32,attn=sdpa,mlm=1.00,mlm-strategy=uniform"

BASE_DIR="/data/share/project/betterbert/gencoder"
CACHE_DIR="$BASE_DIR/hf_cache"

RAW_DATA_DIR="$BASE_DIR/data/raw/xP3/en"
TOKENIZED_DATA_DIR="$BASE_DIR/data/tokenized/xP3-en"

PRETRAINED_CKPT="/data/share/project/betterbert/models/xlm-roberta-xl"
FINETUNED_SAVE_DIR="$BASE_DIR/runs/ft_xlmr_3b_xp3_en_uniform"

if [ ! -d "$TOKENIZED_DATA_DIR" ]; then

    gencoder tokenizer tokenize \
        $RAW_DATA_DIR/*.jsonl \
        --format "json" \
        --tokenizer $PRETRAINED_CKPT \
        --max-length 512 \
        --num-proc 128 \
        --input-key "inputs" \
        --target-key "targets" \
        --save-dir $TOKENIZED_DATA_DIR

fi

accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --rdzv-backend=c10d \
    --main_process_ip=localhost \
    --main_process_port=5050 \
    --mixed_precision bf16 \
    --use_fsdp \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" \
    --fsdp_transformer_layer_cls_to_wrap "XLMRobertaXLLayer" \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_offload_params false \
    --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_forward_prefetch false \
    --fsdp_state_dict_type "SHARDED_STATE_DICT" \
    --no_python \
    gencoder model finetune \
        --train-data $TOKENIZED_DATA_DIR \
        --model-ckpt $PRETRAINED_CKPT \
        --batch-size 8 \
        --grad-accumulation 1 \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --dropout 0.1 \
        --masking-strategy "uniform" \
        --init-mask-prob 1.0 \
        --final-mask-prob 1.0 \
        --save-dir $FINETUNED_SAVE_DIR

gencoder model convert-ckpt \
    --model-type $PRETRAINED_CKPT \
    --model-ckpt "$FINETUNED_SAVE_DIR/checkpoint-125000/pytorch_model_fsdp_0" \
    --save-dir "$FINETUNED_SAVE_DIR/converted-checkpoints/ckpt-125000"

gencoder model inference \
    --model-ckpt "$FINETUNED_SAVE_DIR/converted-checkpoints/ckpt-125000" \
    --batch-size 1 \
    --prompt "Give me a random number." \
    --prompt "Give me a random numerical number." \
    --prompt "Split and simplify the following sentence while retaining its full meaning: as of 2000 , the population was 89,148 .\n Simplified version:"

