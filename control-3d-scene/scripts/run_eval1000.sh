#!/bin/bash
# Regenerate the original Liu baseline on the first 1,000 test windows and evaluate it.
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
CK=/home/ubuntu/genai-final/checkpoints
GEN=/home/ubuntu/genai-final/generated
export NYC_DATA=/home/ubuntu/genai-final/data/city
cd ~/genai-final/control-3d-scene
echo "=== generate, predicted centers $(date -u)"
python launch.py -c configs/nyc_infer.yaml -n nyc_baseline_test1000 \
  --resume_path $CK/nyc_baseline_loc/latest.tar --infer_n 1000
echo "=== generate, true centers $(date -u)"
python launch.py -c configs/nyc_infer.yaml -n nyc_baseline_test1000_gtpos \
  --resume_path $CK/nyc_baseline_loc/latest.tar --infer_n 1000 --use_gt_positions true
cd ~/genai-final/evaluation
for n in nyc_baseline_test1000 nyc_baseline_test1000_gtpos; do
  echo "=== evaluate $n $(date -u)"
  python c2st.py --pred_dir $GEN/$n --name $n
  python evaluate.py --pred_dir $GEN/$n --name $n
done
echo "=== paired comparison $(date -u)"
python evaluate.py --compare results/nyc_baseline_test1000_gtpos results/nyc_baseline_test1000
echo "EVAL1000_DONE $(date -u)"
