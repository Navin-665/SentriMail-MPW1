#!/bin/bash
cd "/c/Users/nan33/OneDrive/Desktop/6th sem projects/SENTRIMAIL-MPW1/sentriformer"; PY=../model_training/.venv/Scripts/python.exe
until grep -q "T5C_COMPLETE\|T5 pipeline gave up" wsp_runs/pipeline/t5c.out 2>/dev/null; do sleep 60; done
for i in 1 2 3; do echo "$(date +%T) START T5a retest attempt $i"; $PY -I -u wsp_runs/t5/t5a_retest.py >> wsp_runs/pipeline/t5a_retest.log 2>&1 && break; [ -f wsp_runs/t5/t5a_retest.json ] && break; sleep 20; done
echo "$(date +%T) RETEST_COMPLETE"
