#!/bin/bash
# NYC-trained CityDreamer: checks, VQGAN (4 h), sampler (8 h), first figures. Stops at a failed gate.
# Run inside tmux:  tmux new -s cdnyc 'bash run_pipeline.sh 2>&1 | tee ~/genai-final/logs/cdnyc_pipeline.log'
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate citydreamer
export NYC_DATA=~/genai-final/data/city
cd ~/genai-final/baselines/citydreamer_nyc
CK=~/genai-final/checkpoints/citydreamer_nyc
GEN=~/genai-final/generated/citydreamer_nyc_dev
mkdir -p $CK $GEN

gate() {   # gate <checkpoint> <key> <min> <label>
  python - "$@" <<'EOF'
import sys, torch
path, key, lo, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
v = torch.load(path, map_location="cpu", weights_only=False)[key]
print(f"GATE {label}: {key} = {v:.4f} (need >= {lo})", flush=True)
sys.exit(0 if v >= lo else 1)
EOF
}

echo "=== 1. VQGAN overfit check (8 windows) $(date -u)"
rm -rf $CK/vq_overfit
python train_vqgan.py --overfit 8 --steps 1500 --batch 16 --val_every 500 --out $CK/vq_overfit
gate $CK/vq_overfit/latest.pth val_acc 0.95 "VQGAN overfit"

echo "=== 2. VQGAN training (4 h) $(date -u)"
python train_vqgan.py --hours 4 --batch 16 --out $CK/vqgan
gate $CK/vqgan/latest.pth val_acc 0.85 "VQGAN validation reconstruction"

echo "=== 3. sampler overfit check (8 windows) $(date -u)"
rm -rf $CK/s_overfit
python train_sampler.py --vqgan $CK/vqgan/latest.pth --overfit 8 --steps 1500 --val_every 500 --out $CK/s_overfit

echo "=== 4. sampler training (8 h) $(date -u)"
python train_sampler.py --vqgan $CK/vqgan/latest.pth --hours 8 --out $CK/sampler

echo "=== 5. first figures (validation windows only) $(date -u)"
python generate.py --ckpt $CK/sampler/latest.pth --uncond 8 --out $GEN
python generate.py --ckpt $CK/sampler/latest.pth --split val --n 4 --viz 4 --out $GEN/val
echo "CDNYC_PIPELINE_DONE $(date -u)"
