#!/bin/bash
# v2 blind verification, 3 workers (resumable per example), then adjudication per batch.
cd "/c/Users/nan33/OneDrive/Desktop/6th sem projects/SENTRIMAIL-MPW1/sentriformer"
PY=../model_training/.venv/Scripts/python.exe
for b in pilot_001 batch_002a batch_002b batch_002c; do
  for i in 1 2 3 4 5; do
    $PY -I synthetic_factory/forge.py verify $b 3 && break; echo "$(date +%T) verify $b failed, retry $i"; sleep 30
  done
  $PY -I synthetic_factory/forge.py adjudicate $b
done
echo "$(date +%T) VERIFY_COMPLETE"
