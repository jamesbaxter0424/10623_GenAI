#!/bin/bash
# Raster diffusion.
#   bash raster_diffusion/run.sh C        one stage: A, B, REG, C, TEST (ALL = A, B, REG, C; stops at the first failed gate)
# REG = the one-pass regression control on 2,000 blocks. Interrupted training: RESUME=1 bash ... C
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; OUT=${STAGE2_OUT:-out}; DO=${1:-}; R=${RESUME:+--resume}
STAGES="A B REG C TEST"
usage() { sed -n '2,/^set -e/p' raster_diffusion/run.sh | grep '^#' | sed 's/^# \{0,1\}//'; exit ${1:-0}; }
[ $# -gt 0 ] || usage
case " $STAGES ALL " in *" $1 "*) ;; *) echo "unknown stage: $1"; usage 1;; esac
want() { [ "$DO" = "$1" ] || { [ "$DO" = ALL ] && [ "$1" != TEST ]; }; }

[ -f $OUT/rule_table.pkl ] || $PY -m common.rule_split --fit
[ -f results/rule_val200/summary.json ] && [ -f $OUT/rule/val200.jsonl ] || { $PY -m common.rule_split --subset val200 --samples 2 --out $OUT/rule/val200.jsonl
  $PY -m common.evaluate --pred $OUT/rule/val200.jsonl --name rule_val200 --subset val200 --boot 0; }

if want A; then
  $PY -m raster_diffusion.train --name rd_overfit8 --subset overfit8 $R --set batch=8 max_steps=4000 val_every=500 patience=99 max_hours=1.0
  $PY -m common.gate A --run $OUT/rd_overfit8 --max_score 0.10
fi
if want B; then
  $PY -m raster_diffusion.train --name rd_pilot --subset pilot2000 $R --set max_steps=20000 max_hours=3.0
  $PY -m common.gate B --run $OUT/rd_pilot
fi
if want REG; then
  $PY -m raster_diffusion.train --name rd_reg_pilot --subset pilot2000 $R --set mode=regression max_steps=20000 max_hours=3.0
fi
if want C; then
  $PY -m raster_diffusion.train --name rd_full --subset train $R
  $PY -m raster_diffusion.tune_instances --ckpt $OUT/rd_full/best.pt
fi
if want TEST; then                       # after `core` is fixed on val200: CORE=1.25 bash ... TEST
  $PY -m raster_diffusion.generate --ckpt $OUT/rd_full/best.pt ${CORE:+--core $CORE} --out $OUT/rd_full/test1000_oracle.jsonl
  $PY -m common.evaluate --pred $OUT/rd_full/test1000_oracle.jsonl --name diffusion_oracle
  for S in ${STAGE1:-}; do               # STAGE1="nyc_c1 nyc_c3": files out/stage1/<name>.pkl
    $PY -m raster_diffusion.generate --ckpt $OUT/rd_full/best.pt ${CORE:+--core $CORE} --input $OUT/stage1/$S.pkl --out $OUT/rd_full/test1000_$S.jsonl
    $PY -m common.evaluate --pred $OUT/rd_full/test1000_$S.jsonl --name diffusion_$S --input $OUT/stage1/$S.pkl
  done
fi
echo RD_RUN_DONE
