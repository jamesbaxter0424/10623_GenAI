#!/bin/bash
# Step 3: overfit 8 training windows (joint GNN+diffusion -> localization post-train
# -> generate the same 8 windows with true and with predicted positions).
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
cd ~/genai-final/control-3d-scene
CK=/home/ubuntu/genai-final/checkpoints
python launch.py -c configs/nyc_train.yaml -n overfit8 \
  --max_samples 8 --repeat 50 --batch_size 8 --epochs 80 --max_hours 0.5 --check_every 1000 --print_every 25
python launch.py -c configs/nyc_posttrain_loc.yaml -n overfit8_loc \
  --resume_path $CK/overfit8/latest.tar --max_samples 8 --repeat 50 --batch_size 8 --epochs 40 --print_every 25
for GT in true false; do
  python launch.py -c configs/nyc_infer.yaml -n overfit8_gen_gtpos_$GT \
    --resume_path $CK/overfit8_loc/latest.tar --infer_split train --infer_subset "" \
    --max_samples 8 --png_n 8 --use_gt_positions $GT
done
echo OVERFIT_DONE
