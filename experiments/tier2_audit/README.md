# `tier2_audit/` has been quarantined

**Moved to `experiments/_quarantine/tier2_audit/` on 2026-08-26.**

This directory is a signpost, not a code location. There are no `.py` files here
and nothing here is importable.

## Why

Every `asym` computed under the old `tier2_audit/` used
`‖(J − Jᵀ)/2‖_F / ‖J‖_F`, **exactly half** the canonical definition in
`theory_and_claims.md`. Several other defects rode along: `negfrac` returned
under the name `negeig`, thirteen unseeded copies of `get_Q`, and a profile
residual that was always `0.00e+00`. The full list is in
`experiments/_quarantine/QUARANTINE.md`.

`FINAL_NUMBERS.md` §1.2 already states that **no number in that document is
taken from these files**.

## If you got here from `FINAL_NUMBERS.md` §9.1

Two data files here are still cited, as *historical* values shown beside a
current measurement and doubled to the canonical convention:

| you wanted | it is now at |
|---|---|
| `tier2_audit/results_tabpfn_5seed.json` | `experiments/_quarantine/tier2_audit/results_tabpfn_5seed.json` |
| `tier2_audit/results_batch1.json` | `experiments/_quarantine/tier2_audit/results_batch1.json` |

Both are safe to cite as historical. Neither may be used to produce a new
number.
