"""Import barrier for quarantined code.

Everything under this directory produced numbers that are known wrong or known
superseded. It is kept for archaeology and for reproducing the retraction
claims in FINAL_NUMBERS.md section 1.3 -- never for producing a new number.

Importing it as a package raises. See QUARANTINE.md in this directory for what
each item is, what is wrong with it, and what supersedes it.

If you genuinely need to run something in here to reproduce a retraction, do it
as a script from inside its own directory, not by importing it into live code.
"""

raise ImportError(
    "experiments._quarantine is quarantined and must not be imported by live "
    "code. Its contents produced numbers that are wrong (halved asym "
    "convention, retracted negfrac, unseeded get_Q, hardcoded thresholds) or "
    "superseded. See experiments/_quarantine/QUARANTINE.md."
)
