#!/bin/bash
# NYC-trained CityDreamer on the 1,000 test windows: generation, evaluation, CNN test, and the
# VQGAN-reconstruction reference row. Run inside tmux:
#   tmux new -s cdtest 'bash run_test.sh 2>&1 | tee ~/genai-final/logs/cdnyc_test.log'
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate citydreamer
export NYC_DATA=~/genai-final/data/city
CKPT=~/genai-final/checkpoints/citydreamer_nyc/sampler/latest.pth
GEN=~/genai-final/generated
EV=~/genai-final/evaluation
cd ~/genai-final/baselines/citydreamer_nyc

echo "=== 1. generation, 1,000 test windows $(date -u)"
python generate.py --ckpt $CKPT --n 1000 --viz 10 --out $GEN/citydreamer_nyc_test1000

echo "=== 2. reference row: VQGAN reconstruction $(date -u)"
python generate.py --ckpt $CKPT --n 1000 --viz 3 --recon --out $GEN/citydreamer_nyc_recon_test1000

cd $EV
for name in citydreamer_nyc_test1000 citydreamer_nyc_recon_test1000; do
  echo "=== 3. evaluate $name $(date -u)"
  python evaluate.py --pred_dir $GEN/$name --name $name
  echo "=== 4. CNN test $name $(date -u)"
  python c2st.py --pred_dir $GEN/$name --name $name
done
echo "CDNYC_TEST_DONE $(date -u)"
