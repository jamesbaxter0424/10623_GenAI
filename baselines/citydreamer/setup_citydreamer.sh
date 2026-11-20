#!/bin/bash
# Set up the CityDreamer layout-generator baseline on the AWS box (no GPU work).
#   - clones CityDreamer into ~/genai-final/third_party/CityDreamer (pinned commit recorded)
#   - clones the genai-final conda env into `citydreamer` and adds easydict
#   - downloads the released Unbounded Layout Generator checkpoint (LayoutGen.pth)
set -e
ROOT=~/genai-final
TP=$ROOT/third_party
mkdir -p $TP $ROOT/checkpoints/citydreamer

if [ ! -d $TP/CityDreamer ]; then
  git clone https://github.com/hzxie/CityDreamer.git $TP/CityDreamer
fi
git -C $TP/CityDreamer rev-parse HEAD > $ROOT/checkpoints/citydreamer/COMMIT
echo "CityDreamer commit: $(cat $ROOT/checkpoints/citydreamer/COMMIT)"

source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^citydreamer "; then
  conda create -y -n citydreamer --clone genai-final
fi
conda activate citydreamer
pip install --no-cache-dir easydict einops

CKPT=$ROOT/checkpoints/citydreamer/LayoutGen.pth
# The download page needs a verification step in a browser, so the file is fetched by hand:
#   https://gateway.infinitescript.com/?f=LayoutGen.pth  ->  $CKPT
if [ ! -s $CKPT ] || head -c 20 $CKPT | grep -q "<"; then
  echo "Missing $CKPT: download LayoutGen.pth from https://gateway.infinitescript.com/?f=LayoutGen.pth"
  exit 1
fi
ls -l $CKPT
python - <<'EOF'
import os, torch, easydict
p = os.path.expanduser("~/genai-final/checkpoints/citydreamer/LayoutGen.pth")
ck = torch.load(p, map_location="cpu", weights_only=False)
print("keys:", list(ck.keys()))
cfg = ck["cfg"]
print("VQGAN:", dict(cfg.NETWORK.VQGAN))
print("SAMPLER:", dict(cfg.NETWORK.SAMPLER))
print("OSM_LAYOUT:", dict(cfg.DATASETS.OSM_LAYOUT))
for k in ("vqae", "sampler"):
    sd = ck[k]
    print(k, len(sd), "tensors, first:", list(sd)[:2])
EOF
echo CITYDREAMER_SETUP_DONE
