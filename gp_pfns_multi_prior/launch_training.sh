#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p /home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs
mkdir -p /home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/checkpoints

# Prevent CPU thrashing by limiting OpenMP/MKL threads per process
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "Launching PFN training runs with thread limits..."

# --- GPU 0 ---
echo "Launching rbf_matern on GPU 0..."
python3 gp_pfns_multi_prior/train_models.py --model_type rbf_matern --gpu 0 --epochs 100 > /home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs/rbf_matern.log 2>&1 &

echo "Launching rbf_periodic on GPU 0..."
python3 gp_pfns_multi_prior/train_models.py --model_type rbf_periodic --gpu 0 --epochs 100 > /home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs/rbf_periodic.log 2>&1 &

# --- GPU 1 ---
echo "Launching matern_periodic on GPU 1..."
python3 gp_pfns_multi_prior/train_models.py --model_type matern_periodic --gpu 1 --epochs 100 > /home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs/matern_periodic.log 2>&1 &

echo "Launching all_three on GPU 1..."
python3 gp_pfns_multi_prior/train_models.py --model_type all_three --gpu 1 --epochs 100 > /home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs/all_three.log 2>&1 &

echo "All training runs launched in the background!"
echo "Check progress by running: tail -f gp_pfns_multi_prior/logs/*.log"
