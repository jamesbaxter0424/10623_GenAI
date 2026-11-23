#!/bin/bash
# Overfitting check for change 1: score 1,000 TRAINING windows with the same generation and
# evaluation as the test set, then compare with nyc_c1_test1000. Waits for run_change1.sh.
#   1 wait for C1_PIPELINE_DONE in c1_pipeline.log (stop if that run failed)
#   2 training subset: same fixed random ordering as the test subset (seed 0), split "train"
#   3 generate 1,000 training windows x 4 samples, true centers
#   4 evaluate.py --split train
# Run:  tmux new -d -s c1t 'bash scripts/run_c1_traineval.sh 2>&1 | tee ~/genai-final/logs/c1_traineval.log'
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
CK=/home/ubuntu/genai-final/checkpoints
GEN=/home/ubuntu/genai-final/generated
LOG=/home/ubuntu/genai-final/logs/c1_pipeline.log
export NYC_DATA=/home/ubuntu/genai-final/data/city
cd ~/genai-final/control-3d-scene

echo "=== 1. waiting for the change-1 pipeline $(date -u)"
until grep -q "C1_PIPELINE_DONE" $LOG; do
  if ! tmux has-session -t c1 2>/dev/null; then
    grep -q "C1_PIPELINE_DONE" $LOG && break
    echo "c1 pipeline stopped without C1_PIPELINE_DONE"; exit 1
  fi
  sleep 120
done

echo "=== 2. training subset $(date -u)"
python - <<'EOF'
import json, numpy as np, pandas as pd
s = pd.read_parquet("/home/ubuntu/genai-final/data/city/samples.parquet")
s = s[(s["split"] == "train") & (~s["ctx_other_split"])]
ids = [int(i) for i in np.random.default_rng(0).permutation(s["sample_id"].to_numpy())]
json.dump({"split": "train", "seed": 0, "sample_ids": ids},
          open("/home/ubuntu/genai-final/data/city/train_subset.json", "w"))
print(len(ids), "training windows, first", ids[:5])
EOF

echo "=== 3. generation, 1,000 training windows, true centers $(date -u)"
python launch.py -c configs/nyc_infer_c1.yaml -n nyc_c1_train1000 \
  --resume_path $CK/nyc_c1/latest.tar --infer_split train \
  --infer_subset /home/ubuntu/genai-final/data/city/train_subset.json \
  --infer_n 1000 --use_gt_positions true

echo "=== 4. evaluation $(date -u)"
cd ~/genai-final/evaluation
python evaluate.py --split train --pred_dir $GEN/nyc_c1_train1000 --name nyc_c1_train1000
echo "C1_TRAINEVAL_DONE $(date -u)"
