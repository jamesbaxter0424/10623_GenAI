#!/bin/bash
# Change 1 (numeric node attributes + street-facing graph) on top of the true-center baseline.
#   1 overfit 8 training windows, generate them with true centers; gate: cell accuracy >= 0.95
#   2 full training, 12 h budget, same settings as the baseline (configs/nyc_train_c1.yaml)
#   3 generate the first 1,000 test windows x 4 samples with true centers
#   4 evaluate (evaluate.py, c2st.py) and compare with the baseline (paired block bootstrap)
# Run:  tmux new -s c1 'bash scripts/run_change1.sh 2>&1 | tee ~/genai-final/logs/c1_pipeline.log'
# Resume step 2 after a crash:  RESUME=1 bash scripts/run_change1.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
CK=/home/ubuntu/genai-final/checkpoints
GEN=/home/ubuntu/genai-final/generated
export NYC_DATA=/home/ubuntu/genai-final/data/city
cd ~/genai-final/control-3d-scene

if [ "${RESUME:-0}" != "1" ]; then
  echo "=== 1. overfit 8 windows $(date -u)"
  python launch.py -c configs/nyc_train_c1.yaml -n c1_overfit8 \
    --max_samples 8 --repeat 50 --batch_size 8 --epochs 80 --max_hours 0.5 --check_every 1000 --print_every 25
  python launch.py -c configs/nyc_infer_c1.yaml -n c1_overfit8_gen \
    --resume_path $CK/c1_overfit8/latest.tar --infer_split train --infer_subset "" \
    --max_samples 8 --png_n 8 --use_gt_positions true
  python tools/nyc_quick_eval.py $GEN/c1_overfit8_gen/Generated | tee $GEN/c1_overfit8_metrics.txt
  ACC=$(grep -m1 "cell_acc" $GEN/c1_overfit8_metrics.txt | awk '{print $2}')
  python -c "import sys; a=float('$ACC'); print(f'GATE overfit cell_acc = {a:.3f} (need >= 0.95)'); sys.exit(0 if a >= 0.95 else 1)"
fi

echo "=== 2. training (12 h) $(date -u)"
EXTRA=""
if [ "${RESUME:-0}" = "1" ]; then
  EXTRA="--resume true --resume_optimizer true --resume_path $CK/nyc_c1/latest.tar"
fi
python launch.py -c configs/nyc_train_c1.yaml -n nyc_c1 $EXTRA
echo C1_TRAIN_DONE

echo "=== 3. generation, 1,000 test windows, true centers $(date -u)"
python launch.py -c configs/nyc_infer_c1.yaml -n nyc_c1_test1000 \
  --resume_path $CK/nyc_c1/latest.tar --infer_n 1000 --use_gt_positions true

echo "=== 4. evaluation $(date -u)"
cd ~/genai-final/evaluation
python evaluate.py --pred_dir $GEN/nyc_c1_test1000 --name nyc_c1_test1000
python c2st.py --pred_dir $GEN/nyc_c1_test1000 --name nyc_c1_test1000
python evaluate.py --compare results/nyc_c1_test1000 results/nyc_baseline_test1000_gtpos
echo "C1_PIPELINE_DONE $(date -u)"
