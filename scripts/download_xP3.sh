
BASE_DIR="/data/share/project/betterbert/gencoder"
CACHE_DIR="$BASE_DIR/hf_cache"
RAW_DATA_DIR="$BASE_DIR/data/raw/xP3"

gencoder data hf-download \
    --repo-id "bigscience/xP3" \
    --pattern "**/xp3_*.jsonl" \
    --cache-dir $CACHE_DIR \
    --save-dir $RAW_DATA_DIR
