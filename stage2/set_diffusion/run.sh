#!/bin/bash
# Set diffusion.
#   bash set_diffusion/run.sh C        one stage: A, B, C, VAL, TEST (ALL = A, B, C; stops at the first failed gate)
# C = final training on all blocks.   VAL = predictions on val200 for the figure.   TEST = test1000, once.
# A and B are run by James. Interrupted training: RESUME=1 bash set_diffusion/run.sh C
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; OUT=${STAGE2_OUT:-out}; DO=${1:-}; R=${RESUME:+--resume}
STAGES="A B C VAL TEST"
usage() { sed -n '2,/^set -e/p' set_diffusion/run.sh | grep '^#' | sed 's/^# \{0,1\}//'; exit ${1:-0}; }
[ $# -gt 0 ] || usage
case " $STAGES ALL " in *" $1 "*) ;; *) echo "unknown stage: $1"; usage 1;; esac
want() { [ "$DO" = "$1" ] || { [ "$DO" = ALL ] && [ "$1" != VAL ] && [ "$1" != TEST ]; }; }

[ -f $OUT/rule_table.pkl ] || $PY -m common.rule_split --fit
[ -f results/rule_val200/summary.json ] && [ -f $OUT/rule/val200.jsonl ] || { $PY -m common.rule_split --subset val200 --samples 2 --out $OUT/rule/val200.jsonl
  $PY -m common.evaluate --pred $OUT/rule/val200.jsonl --name rule_val200 --subset val200 --boot 0; }

if want A; then
  $PY -m set_diffusion.train --name sd_overfit8 --subset overfit8 $R --set max_steps=8000 val_every=1000 patience=99 warmup=200 max_hours=1.0
  $PY -m common.gate A --run $OUT/sd_overfit8 --max_score 0.15
fi
if want B; then
  $PY -m set_diffusion.train --name sd_pilot --subset pilot2000 $R --set max_steps=60000 max_hours=4.0 patience=99   # long on purpose: is the training stable?
  $PY -m common.gate B --run $OUT/sd_pilot
fi
if want C; then
  $PY -m set_diffusion.train --name sd_full --subset train $R
fi
if want VAL; then
  $PY -m set_diffusion.generate --ckpt $OUT/sd_full/best.pt --subset val200 --samples 1 --out $OUT/sd_full/val200.jsonl
  $PY -m common.evaluate --pred $OUT/sd_full/val200.jsonl --name sd_full_val200 --subset val200
fi
if want TEST; then
  for S in ${STAGE1:-}; do [ -f $OUT/stage1/$S.pkl ] || { echo "missing $OUT/stage1/$S.pkl (ask James)"; exit 1; }; done
  [ -z "$(ls -d results/set_oracle 2>/dev/null)" ] || [ -n "$FORCE" ] || { echo "results/set_oracle exists: test1000 is scored once. FORCE=1 only to finish an interrupted TEST."; exit 1; }
  $PY -m set_diffusion.generate --ckpt $OUT/sd_full/best.pt --out $OUT/sd_full/test1000_oracle.jsonl
  $PY -m common.evaluate --pred $OUT/sd_full/test1000_oracle.jsonl --name set_oracle
  $PY -m set_diffusion.generate --ckpt $OUT/sd_full/best.pt --no_trim --samples 1 --out $OUT/sd_full/test1000_oracle_raw.jsonl
  $PY -m common.evaluate --pred $OUT/sd_full/test1000_oracle_raw.jsonl --name set_oracle_raw
  for S in ${STAGE1:-}; do               # STAGE1="nyc_c1 nyc_c3": files out/stage1/<name>.pkl
    $PY -m set_diffusion.generate --ckpt $OUT/sd_full/best.pt --input $OUT/stage1/$S.pkl --out $OUT/sd_full/test1000_$S.jsonl
    $PY -m common.evaluate --pred $OUT/sd_full/test1000_$S.jsonl --name set_$S --input $OUT/stage1/$S.pkl
  done
fi
echo SD_RUN_DONE
