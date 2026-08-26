#!/bin/bash
nohup env CUDA_VISIBLE_DEVICES=0 python3 experiments/tier2_audit/audit_phase2_curvature.py > experiments/tier2_audit/curve_out.txt 2>&1 &
echo "Started background process."
