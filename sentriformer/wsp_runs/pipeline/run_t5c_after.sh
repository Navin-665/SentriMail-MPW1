#!/bin/bash
# waits for the T5 pipeline, then runs T5c stage A and (if a lambda is eligible) stage B. Each stage retried up to 3 times.
cd "/c/Users/nan33/OneDrive/Desktop/6th sem projects/SENTRIMAIL-MPW1/sentriformer"; PY=../model_training/.venv/Scripts/python.exe
until grep -q "PIPELINE_COMPLETE\|GIVEUP" wsp_runs/pipeline/t5_pipeline.out 2>/dev/null; do sleep 60; done
grep -q GIVEUP wsp_runs/pipeline/t5_pipeline.out && { echo "T5 pipeline gave up; T5c not run"; exit 1; }
for st in stageA stageB; do
  [ -f wsp_runs/t5/t5c_results/$st.json ] && continue
  if [ $st = stageB ] && grep -q '"selected_lambda": null' wsp_runs/t5/t5c_results/stageA.json; then echo "T5c rejected at stage A"; break; fi
  for i in 1 2 3; do echo "$(date +%T) START T5c $st attempt $i"; $PY -I -u wsp_runs/t5/t5c_mmi.py $st >> wsp_runs/pipeline/t5c_$st.log 2>&1 && break; sleep 20; done
  echo "$(date +%T) END T5c $st"
done
echo "$(date +%T) T5C_COMPLETE"
