#!/bin/bash
# GlobalMapper retrained on NYC (the "block shape only" baseline), official code and settings.
#   0 conversion is done before (prep_nyc.py -> ~/genai-final/data/gm_nyc)
#   1 overfit 8 training blocks; gate: validation (= the same 8) geometry loss < 10% of epoch 0
#   2 training on the NYC train split, validation on the val split; 250 epochs or 12 h
#   3 generation of the target blocks of the first 1,000 test windows x 4 samples:
#     random (release test.py), prior (paper's sparse prior, k = 5), recon (reference)
#   4 evaluation (evaluate.py) and figures
# Run:  tmux new -d -s gm 'bash run_pipeline.sh 2>&1 | tee ~/genai-final/logs/gm_pipeline.log'
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate genai-final
H=/home/ubuntu/genai-final
GMC=$H/third_party/GlobalMapper
DATA=$H/data/gm_nyc
CK=$H/checkpoints
GEN=$H/generated
export NYC_DATA=$H/data/city
cd $H/baselines/globalmapper_nyc

if [ "${SKIP_GATE:-0}" != "1" ]; then
  echo "=== 1. overfit 8 blocks $(date -u)"
  rm -rf $CK/gm_overfit8
  python train_nyc.py --data $DATA --gm $GMC --out $CK/gm_overfit8 --max_samples 8 --max_epochs 400 | tail -3
  python - <<EOF
import pandas as pd, sys
h = pd.read_csv("$CK/gm_overfit8/training_history.csv")
r = h.val_geo_loss.iloc[-1] / h.val_geo_loss.iloc[0]
print(f"GATE overfit: geometry loss {h.val_geo_loss.iloc[0]:.1f} -> {h.val_geo_loss.iloc[-1]:.1f} "
      f"(ratio {r:.3f}, need < 0.10); exist acc {h.val_acc.iloc[-1]:.3f}")
sys.exit(0 if r < 0.10 else 1)
EOF
fi

echo "=== 2. training $(date -u)"
python train_nyc.py --data $DATA --gm $GMC --out $CK/globalmapper_nyc --max_epochs 250 --max_hours 12
echo GM_TRAIN_DONE

for MODE in random prior recon; do
  echo "=== 3. generation $MODE $(date -u)"
  python gen_nyc.py --data $DATA --gm $GMC --ckpt $CK/globalmapper_nyc/val_least_loss_all.pth \
    --city $NYC_DATA --mode $MODE --out $GEN/globalmapper_nyc/$MODE
done

cd $H/evaluation
for MODE in random prior recon; do
  echo "=== 4. evaluation $MODE $(date -u)"
  python evaluate.py --pred_dir $GEN/globalmapper_nyc/$MODE --name globalmapper_nyc_${MODE}_test1000
done
python evaluate.py --compare results/globalmapper_nyc_prior_test1000 results/globalmapper_nyc_random_test1000
cd $H/baselines/globalmapper_nyc
python viz_gen.py --gen $GEN/globalmapper_nyc --modes random prior recon --city $NYC_DATA \
  --out $GEN/globalmapper_nyc/test_examples.png
echo "GM_PIPELINE_DONE $(date -u)"
