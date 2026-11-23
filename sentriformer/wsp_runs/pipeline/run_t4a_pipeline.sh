#!/bin/bash
# T4a resume -> evals, strictly sequential. Each step retried (max 5) and marked .done; reruns skip finished steps.
cd "/c/Users/nan33/OneDrive/Desktop/6th sem projects/SENTRIMAIL-MPW1/sentriformer"
PY=../model_training/.venv/Scripts/python.exe; P=wsp_runs/pipeline
COMMON="--init wsp_runs/FINAL_T3/sentriformer_t3.pt --head_warmup 0 --lr_heads 3e-4 --max_steps 18000 --max_minutes 150 --eval_every 1000 --n_cat_train 500000 --bs_cat 48 --bs_csat 16 --bs_sent 32 --bs_dial 32 --csat_every 3 --urg_every 4 --dial_ctx_len 256"
step() { # name, partial-output-dir-to-clear-on-retry, command...
  n=$1; d=$2; shift 2
  [ -f $P/$n.done ] && { echo "$(date +%T) skip $n"; return 0; }
  for i in 1 2 3 4 5; do
    [ -n "$d" ] && rm -rf "$d"
    echo "$(date +%T) START $n attempt $i"
    "$@" >> $P/$n.log 2>&1 && { touch $P/$n.done; echo "$(date +%T) DONE $n"; return 0; }
    echo "$(date +%T) FAIL $n rc=$?"; sleep 20
  done; echo "$(date +%T) GIVEUP $n"; return 1
}
step train_T4a "" $PY -I wsp/train_v2.py --run T4a $COMMON --dial_train wsp_runs/t4/data/multi_train.parquet --dial_val wsp_runs/t4/data/multi_val.parquet || exit 1
step t4_T3      wsp_runs/t4/eval/T3_val       $PY -I wsp/eval_t4.py --ckpt wsp_runs/FINAL_T3/sentriformer_t3.pt --name T3 --fmt raw
step t4_ctrl    wsp_runs/t4/eval/T4a_ctrl_val $PY -I wsp/eval_t4.py --ckpt wsp_runs/train/T4a_ctrl/model_best.pt --name T4a_ctrl --fmt ctrl
step t4_T4a     wsp_runs/t4/eval/T4a_val      $PY -I wsp/eval_t4.py --ckpt wsp_runs/train/T4a/model_best.pt --name T4a --fmt multi
step full_ctrl  wsp_runs/eval_full/T4a_ctrl_val $PY -I wsp/eval_full.py --ckpt wsp_runs/train/T4a_ctrl/model_best.pt --name T4a_ctrl --split val
step full_T4a   wsp_runs/eval_full/T4a_val      $PY -I wsp/eval_full.py --ckpt wsp_runs/train/T4a/model_best.pt --name T4a --split val
step acc_ctrl "" $PY -I wsp/accept_training.py T4a_ctrl
step acc_T4a  "" $PY -I wsp/accept_training.py T4a
echo "$(date +%T) PIPELINE_COMPLETE"
