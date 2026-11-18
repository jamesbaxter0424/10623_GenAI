#!/bin/bash
# Steps 4-5: original Liu model on NYC.
#   4a joint GNN + diffusion training (12 h budget)   -> checkpoints/nyc_baseline
#   4b localization-head post-training (1 h budget)   -> checkpoints/nyc_baseline_loc
#   5  generate 200 test windows x 4 samples, predicted positions (= the baseline)
#      and true positions (diagnostic), then quick metrics.
# Resume 4a after a crash:  RESUME=1 bash scripts/run_baseline.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
cd ~/genai-final/control-3d-scene
CK=/home/ubuntu/genai-final/checkpoints
GEN=/home/ubuntu/genai-final/generated
EXTRA=""
if [ "${RESUME:-0}" = "1" ]; then
  EXTRA="--resume true --resume_optimizer true --resume_path $CK/nyc_baseline/latest.tar"
fi
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  python launch.py -c configs/nyc_train.yaml -n nyc_baseline $EXTRA
  echo STEP4A_DONE
fi
python launch.py -c configs/nyc_posttrain_loc.yaml -n nyc_baseline_loc \
  --resume_path $CK/nyc_baseline/latest.tar
echo STEP4B_DONE
python launch.py -c configs/nyc_infer.yaml -n nyc_baseline_test200 \
  --resume_path $CK/nyc_baseline_loc/latest.tar --infer_n 200
python launch.py -c configs/nyc_infer.yaml -n nyc_baseline_test200_gtpos \
  --resume_path $CK/nyc_baseline_loc/latest.tar --infer_n 200 --use_gt_positions true
python tools/nyc_quick_eval.py $GEN/nyc_baseline_test200/Generated $GEN/nyc_baseline_test200_gtpos/Generated \
  | tee $GEN/nyc_baseline_test200_metrics.txt
echo BASELINE_DONE
