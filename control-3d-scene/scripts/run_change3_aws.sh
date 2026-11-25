#!/bin/bash
# Change 3 on AWS g5.xlarge (A10G), used because the Delta A100 queue did not start.
# The overfit gate already passed on Delta (cell accuracy 0.998, logs in progress/step13_change3).
#   0 pre-training checks   1 train 55 epochs (latest.tar after every epoch; RESUME=1 resumes)
#   2 generate the first 1,000 test windows x 4 samples, true centers   3 evaluate and compare
# Run:  tmux new -d -s c3 'bash scripts/run_change3_aws.sh 2>&1 | tee -a ~/genai-final/logs/c3_pipeline.log'
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
CK=/home/ubuntu/genai-final/checkpoints
GEN=/home/ubuntu/genai-final/generated
export NYC_DATA=/home/ubuntu/genai-final/data/city
cd ~/genai-final/control-3d-scene

EXTRA=""
if [ "${RESUME:-0}" = "1" ]; then
  EXTRA="--resume true --resume_optimizer true --resume_path $CK/nyc_c3/latest.tar"
else
  echo "=== 0. checks $(date -u)"
  python tools/nyc_c3_check.py --root $NYC_DATA
fi

echo "=== 1. training (55 epochs) $(date -u)"
python launch.py -c configs/nyc_train_c3_aws.yaml -n nyc_c3 $EXTRA
echo "C3_TRAIN_DONE $(date -u)"

echo "=== 2. generation, 1,000 test windows, true centers $(date -u)"
python launch.py -c configs/nyc_infer_c3_aws.yaml -n nyc_c3_test1000 \
  --resume_path $CK/nyc_c3/latest.tar --infer_n 1000 --use_gt_positions true

echo "=== 3. evaluation $(date -u)"
cd ~/genai-final/evaluation
python evaluate.py --pred_dir $GEN/nyc_c3_test1000 --name nyc_c3_test1000
python c2st.py --pred_dir $GEN/nyc_c3_test1000 --name nyc_c3_test1000
python evaluate.py --compare results/nyc_c3_test1000 results/nyc_c1_test1000
python evaluate.py --compare results/nyc_c3_test1000 results/nyc_baseline_test1000_gtpos
echo "C3_PIPELINE_DONE $(date -u)"
