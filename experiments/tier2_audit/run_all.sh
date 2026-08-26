#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -u experiments/tier2_audit/item2.py > experiments/tier2_audit/item2_log.txt 2>&1
python3 -u experiments/tier2_audit/item3.py > experiments/tier2_audit/item3_log.txt 2>&1
