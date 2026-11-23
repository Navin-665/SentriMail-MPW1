#!/bin/bash
# T6: wait for pmi scores -> build -> T6 + T6_ctrl (from T5a) -> eval_full -> frozen compare -> (PASS) regression + behaviour suite.
# Sequential, each step retried (max 5) and marked .done; resumable.
cd "/c/Users/nan33/OneDrive/Desktop/6th sem projects/SENTRIMAIL-MPW1/sentriformer"
PY=../model_training/.venv/Scripts/python.exe; P=wsp_runs/pipeline
COMMON="--init wsp_runs/FINAL_T5a/sentriformer_t5a.pt --head_warmup 0 --lr_heads 3e-4 --max_steps 10000 --max_minutes 150 --eval_every 1000 --n_cat_train 500000 --bs_cat 48 --bs_csat 16 --bs_sent 32 --bs_dial 32 --csat_every 3 --urg_every 4 --dial_ctx_len 256 --dial_val wsp_runs/t4/data/multi_val.parquet"
step() { n=$1; d=$2; shift 2
  [ -f $P/$n.done ] && { echo "$(date +%T) skip $n"; return 0; }
  for i in 1 2 3 4 5; do [ -n "$d" ] && rm -rf "$d"; echo "$(date +%T) START $n attempt $i"
    "$@" >> $P/$n.log 2>&1 && { touch $P/$n.done; echo "$(date +%T) DONE $n"; return 0; }; echo "$(date +%T) FAIL $n rc=$?"; sleep 20; done
  echo "$(date +%T) GIVEUP $n"; return 1; }
while [ ! -f wsp_runs/t6/t6_pmi_scores.parquet ]; do
  grep -q Traceback $P/t6_pmi.log && { step pmi_T6 "" $PY -I -u wsp_runs/t6/t6_pmi.py || exit 1; }
  sleep 60; done
step build_T6 "" $PY -I wsp_runs/t6/build_t6.py || exit 1
step train_T6      "" $PY -I wsp/train_v2.py --run T6      $COMMON --dial_train wsp_runs/t4/data/t6_train.parquet || exit 1
step train_T6_ctrl "" $PY -I wsp/train_v2.py --run T6_ctrl $COMMON --dial_train wsp_runs/t4/data/t5a_train.parquet || exit 1
step full_T6      wsp_runs/eval_full/T6_val      $PY -I wsp/eval_full.py --ckpt wsp_runs/train/T6/model_best.pt --name T6 --split val || exit 1
step compare_T6 "" $PY -I -u wsp_runs/t6/t6_compare.py || exit 1
if [ -d wsp_runs/FINAL_T6 ]; then
  step regress_T6 "" $PY -I wsp/regression_tests.py wsp_runs/FINAL_T6/sentriformer_t6.pt wsp_runs/FINAL_T6/regression_tests.json --trained wsp_runs/eval_full/T6_val/preds_category_len512.npz
  step behave_T6 "" $PY -I wsp/behavior_suite.py wsp_runs/FINAL_T6/sentriformer_t6.pt wsp_runs/FINAL_T6/behavior_suite
fi
echo "$(date +%T) PIPELINE_COMPLETE"
