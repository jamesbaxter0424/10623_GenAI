#!/bin/bash
# GlobalMapper driven by the map. Needs GM_REPO, GM_CKPT, GM_DATA (README, step 1).
#   bash globalmapper_map/run.sh PREP     one stage: PREP, A, B, GAUSS, C, TUNE, TEST (ALL = every stage except TUNE and TEST)
# C    = final training on all blocks.   TUNE = temperatures on val200.   TEST = test1000, once.
# A, B, GAUSS were run by James (Nov 24-25). Interrupted training: RESUME=1 bash ... C
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; OUT=${STAGE2_OUT:-out}; DO=${1:-}; R=${RESUME:+--resume}
STAGES="PREP A B GAUSS C TUNE TEST"
usage() { sed -n '2,/^set -e/p' globalmapper_map/run.sh | grep '^#' | sed 's/^# \{0,1\}//'; exit ${1:-0}; }
[ $# -gt 0 ] || usage
case " $STAGES ALL " in *" $1 "*) ;; *) echo "unknown stage: $1"; usage 1;; esac
want() { [ "$DO" = "$1" ] || { [ "$DO" = ALL ] && [ "$1" != TUNE ] && [ "$1" != TEST ]; }; }

[ -f $OUT/rule_table.pkl ] || $PY -m common.rule_split --fit
[ -f results/rule_val200/summary.json ] && [ -f $OUT/rule/val200.jsonl ] || { $PY -m common.rule_split --subset val200 --samples 2 --out $OUT/rule/val200.jsonl
  $PY -m common.evaluate --pred $OUT/rule/val200.jsonl --name rule_val200 --subset val200 --boot 0; }

if want PREP; then
  $PY -m globalmapper_map.heights --fit
  $PY -m globalmapper_map.extract_latents
  $PY -m globalmapper_map.quantize --levels 10 20 30 | tee $OUT/gm_latents/quantize.txt
fi
if want A; then                          # the ceiling of this decoder is about 0.18, so the gate is 0.22
  $PY -m globalmapper_map.train --name gm_overfit8 --subset overfit8 $R --set batch=32 max_steps=3000 val_every=500 patience=99 warmup=100 max_hours=1.0
  $PY -m common.gate A --run $OUT/gm_overfit8 --max_score 0.22
fi
if want B; then
  $PY -m globalmapper_map.train --name gm_pilot --subset pilot2000 $R --set max_steps=15000 max_hours=2.0
  $PY -m common.gate B --run $OUT/gm_pilot
fi
if want GAUSS; then
  $PY -m globalmapper_map.train --name gm_gauss_pilot --subset pilot2000 $R --set head=gaussian max_steps=15000 max_hours=2.0
fi
if want C; then
  $PY -m globalmapper_map.train --name gm_full --subset train $R
fi
if want TUNE; then                       # val200 only: pick one of these temperatures
  for TT in 0.8 1.0; do
    $PY -m globalmapper_map.generate --ckpt $OUT/gm_full/best.pt --subset val200 --samples 1 --temperature $TT --out $OUT/gm_full/val200_T$TT.jsonl
    $PY -m common.evaluate --pred $OUT/gm_full/val200_T$TT.jsonl --name gm_full_val200_T$TT --subset val200
  done
fi
if want TEST; then                       # after the temperature is fixed on val200: TEMP=0.8 bash ... TEST
  for S in ${STAGE1:-}; do [ -f $OUT/stage1/$S.pkl ] || { echo "missing $OUT/stage1/$S.pkl (ask James)"; exit 1; }; done
  [ -z "$(ls -d results/globalmapper_oracle 2>/dev/null)" ] || [ -n "$FORCE" ] || { echo "results/globalmapper_oracle exists: test1000 is scored once. FORCE=1 only to finish an interrupted TEST."; exit 1; }
  $PY -m globalmapper_map.generate --ckpt $OUT/gm_full/best.pt ${TEMP:+--temperature $TEMP} --out $OUT/gm_full/test1000_oracle.jsonl
  $PY -m common.evaluate --pred $OUT/gm_full/test1000_oracle.jsonl --name globalmapper_oracle
  $PY -m globalmapper_map.generate --ckpt $OUT/gm_full/best.pt ${TEMP:+--temperature $TEMP} --no_trim --samples 1 --out $OUT/gm_full/test1000_oracle_raw.jsonl
  $PY -m common.evaluate --pred $OUT/gm_full/test1000_oracle_raw.jsonl --name globalmapper_oracle_raw
  for S in ${STAGE1:-}; do               # STAGE1="nyc_c1 nyc_c3": files out/stage1/<name>.pkl
    $PY -m globalmapper_map.generate --ckpt $OUT/gm_full/best.pt ${TEMP:+--temperature $TEMP} --input $OUT/stage1/$S.pkl --out $OUT/gm_full/test1000_$S.jsonl
    $PY -m common.evaluate --pred $OUT/gm_full/test1000_$S.jsonl --name globalmapper_$S --input $OUT/stage1/$S.pkl
  done
fi
echo GM_RUN_DONE
