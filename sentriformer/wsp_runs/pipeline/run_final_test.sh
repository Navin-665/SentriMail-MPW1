#!/bin/bash
cd "/c/Users/nan33/OneDrive/Desktop/6th sem projects/SENTRIMAIL-MPW1/sentriformer"; PY=../model_training/.venv/Scripts/python.exe; P=wsp_runs/pipeline
run() { n=$1; shift; [ -f $P/$n.done ] && return 0; echo "$(date +%T) START $n"; "$@" >> $P/$n.log 2>&1 && { touch $P/$n.done; echo "$(date +%T) DONE $n"; } || echo "$(date +%T) FAIL $n"; }
run test_full_T4d $PY -I wsp/eval_full.py --ckpt wsp_runs/train/T4d/model_best.pt --name T4d --split test
run test_t4_T3    $PY -I wsp/eval_t4.py --ckpt wsp_runs/FINAL_T3/sentriformer_t3.pt --name T3 --fmt raw --split test
run test_t4_T4d   $PY -I wsp/eval_t4.py --ckpt wsp_runs/train/T4d/model_best.pt --name T4d --fmt multi --split test
echo "$(date +%T) FINAL_TEST_COMPLETE"
