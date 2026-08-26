> **PARTIALLY SUPERSEDED — read before citing.** Produced by
> `e1_preprocessing_confounds.py`, which builds `Q` with a local **unseeded**
> `get_Q` (line 12), so none of these numbers is reproducible.
>
> | line | status |
> |---|---|
> | E1.1 preprocessing identification | **superseded** by `FINAL_NUMBERS.md` §2.7, which measures all three models plus two controls at five shift/scale amplitudes |
> | E1.2 (a) and (b) projection | **superseded** by `FINAL_NUMBERS.md` §2.2 (reduced floor) and §2.9 (ambient floor) |
> | E1.3 ensemble-of-one | **only record in the tree.** Not in FINAL_NUMBERS. Unseeded `Q`. |
> | E1.5 Euler defect | **only record in the tree.** Not in FINAL_NUMBERS. Unseeded `Q`. |
>
> E1.3 and E1.5 should be re-run through the seeded instrument before the paper
> cites either. See `experiments/_quarantine/QUARANTINE.md`, "Suspect but
> deliberately retained".

# Tier 1: Preprocessing Confounds Report

- **E1.1 Preprocessing Identification**: PASS (shift_err=2.67e-07, scale_err=3.64e-07)
- **E1.3 Ensemble-of-One**: PASS (n_estimators=1)
- **E1.5 Euler Defect**: e_ratio = 4.1287e-04
- **E1.2 (a) Projection on GP**: PASS (raw_asym=0.0162, red_asym=1.0463e-11)
- **E1.2 (b) Projection on Imitator**: PASS (red_negeig=3.3359e-01 >= 1.00e-06)
