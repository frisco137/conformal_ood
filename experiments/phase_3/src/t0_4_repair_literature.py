"""T0.4 -- repair the literature extraction artifact. CPU, no model calls.

The plan (section 2, T0.4) describes the damage as: camp `a` claims 12 files and
lists 10 while 20 `##` sections follow; camp `c` has no manifest and no date;
**camp `b` is absent entirely**.

TWO OF THOSE THREE ARE WRONG, and this script establishes what is actually true
before changing anything. Findings are written to
`results/t0_4_literature_audit.json` and the repaired file to
`literature/literature_extraction_all_REPAIRED.md`. The original is not modified.

WHAT IS ACTUALLY WRONG
----------------------
1. Camp `b` is NOT absent. It is at line 772. Its header is `Camp directory: b`
   -- plain text, not a `#` heading -- which is why a `^#` scan misses it. That
   malformed header is also why camp `a`'s block appears to contain 20 sections:
   the `^#`-delimited region starting at line 1 runs through camp b's papers.
2. Camp `c` DOES have a manifest and a date (lines 1617-1632). What it has
   instead is two papers that belong to other camps.
3. The real defects are three manifest miscounts and two misfiled papers.

Ground truth is the directory listing: `literature/{a,b,c}/*.md`, 11 files each,
33 total.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LIT = REPO / "literature"
SRC = LIT / "literature_extraction_all.md"
OUT = LIT / "literature_extraction_all_REPAIRED.md"
RESULTS = HERE.parent / "results" / "t0_4_literature_audit.json"


def norm(s):
    """Loose title key: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


#: Two camp-a papers whose PRINTED title differs from their filename, so neither
#: exact nor prefix matching finds them. Mapped explicitly rather than by
#: loosening the matcher, which would risk silent mis-assignment elsewhere.
TITLE_ALIASES = {
    norm("Transformers Learn In-Context by Gradient Descent"):
        norm("Transformers learn in-context learning by gradient descent"),
    norm("What Can Transformers Learn In-Context? A Case Study of Simple Function Classes"):
        norm("What Can Transformers Learn In-Context Case Studies in Learning Function Classes"),
}


def main():
    lines = SRC.read_text().split("\n")
    audit = {"source": str(SRC), "n_lines": len(lines)}

    # ---- ground truth from the directories
    truth = {c: sorted(p.stem for p in (LIT / c).glob("*.md")) for c in "abc"}
    audit["dir_counts"] = {c: len(v) for c, v in truth.items()}
    audit["dir_total"] = sum(len(v) for v in truth.values())
    print("GROUND TRUTH (directory listing)")
    for c, v in truth.items():
        print(f"  camp {c}: {len(v)} files")
    print(f"  total: {audit['dir_total']}")

    # ---- locate the three camp blocks, including the malformed one
    starts = []
    for i, l in enumerate(lines):
        if re.match(r"^#\s*Camp directory", l) or re.match(r"^Camp directory:", l):
            starts.append(i)
    starts.append(len(lines))
    audit["camp_block_starts_1indexed"] = [s + 1 for s in starts[:-1]]
    hash_only = sum(1 for l in lines if re.match(r"^#\s*Camp directory", l))
    audit["camp_headers_found_by_hash_scan"] = hash_only
    print(f"\nCAMP BLOCKS found at lines {audit['camp_block_starts_1indexed']}"
          f"  (a plain `^#` scan finds only {hash_only} of them)")

    # ---- sections, and which block each falls in
    secs = [(i, lines[i][3:].strip()) for i in range(len(lines)) if lines[i].startswith("## ")]
    blocks = []
    for k in range(len(starts) - 1):
        lo, hi = starts[k], starts[k + 1]
        blk = [(i, t) for i, t in secs if lo < i < hi]
        blocks.append({"start": lo, "end": hi, "sections": blk})
    for k, b in enumerate(blocks):
        print(f"  block {k} (line {b['start']+1}): {len(b['sections'])} '##' sections")

    # ---- assign every section to its TRUE camp by matching the directory listing
    key2camp = {}
    for c, files in truth.items():
        for f in files:
            key2camp[norm(f)] = c

    assign, unmatched = [], []
    for k, b in enumerate(blocks):
        declared = "abc"[k]
        for i, title in b["sections"]:
            nk = TITLE_ALIASES.get(norm(title), norm(title))
            hit = key2camp.get(nk)
            if hit is None:                       # fall back to fuzzy containment
                cands = [(kk, cc) for kk, cc in key2camp.items()
                         if nk[:40] and (nk[:40] in kk or kk[:40] in nk)]
                hit = cands[0][1] if len(cands) == 1 else None
            assign.append({"line": i + 1, "title": title,
                           "declared_camp": declared, "true_camp": hit,
                           "misfiled": bool(hit and hit != declared)})
            if hit is None:
                unmatched.append(title)

    mis = [a for a in assign if a["misfiled"]]
    audit["n_sections"] = len(assign)
    audit["unmatched_titles"] = unmatched
    audit["misfiled"] = mis
    print(f"\nSECTIONS: {len(assign)} total, {len(unmatched)} unmatched to any directory file")
    print(f"MISFILED: {len(mis)}")
    for m in mis:
        print(f"  line {m['line']:>5}  declared {m['declared_camp']} -> truly "
              f"{m['true_camp']}   {m['title'][:66]}")

    # ---- manifest claims vs reality
    manifest = {}
    for k, b in enumerate(blocks):
        c = "abc"[k]
        head = "\n".join(lines[b["start"]:b["start"] + 20])
        num = re.search(r"(\d+)\s*(?:files|\.md files found in it)", head) \
            or re.search(r"files found in it[^0-9]*(\d+)", head) \
            or re.search(r"Number of files:\s*(\d+)", head)
        listed = len(re.findall(r"^-\s+.+\.md\s*$", head, flags=re.M))
        manifest[c] = {"claimed": int(num.group(1)) if num else None,
                       "listed": listed,
                       "sections_in_block": len(b["sections"]),
                       "true_count": len(truth[c])}
    audit["manifest"] = manifest
    print(f"\n  {'camp':<6}{'claims':>8}{'lists':>7}{'sections':>10}{'TRUE':>6}")
    for c, m in manifest.items():
        flag = "" if m["claimed"] == m["true_count"] == m["listed"] else "   <- wrong"
        print(f"  {c:<6}{str(m['claimed']):>8}{m['listed']:>7}"
              f"{m['sections_in_block']:>10}{m['true_count']:>6}{flag}")

    # ---- write the repaired file: three well-formed camps, sections regrouped
    if unmatched:
        raise SystemExit(f"REFUSING TO WRITE: {len(unmatched)} sections could not be "
                         f"assigned to a camp: {unmatched}. Dropping them silently is "
                         f"exactly the failure this script exists to fix.")
    body = {}
    for a in assign:
        body.setdefault(a["true_camp"], []).append(a)
    sec_start = {i: None for i, _ in secs}
    # A section ends at the next section OR at the end of its camp block,
    # whichever comes first. Without the block bound, the last section of a camp
    # swallows the next camp's header, which is how the first attempt produced 4
    # camp headers in a 3-camp file.
    block_end = {}
    for b in blocks:
        for i, _ in b["sections"]:
            block_end[i] = b["end"]
    idxs = [i for i, _ in secs]
    span = {}
    for j, i in enumerate(idxs):
        nxt = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        span[i] = (i, min(nxt, block_end[i]))

    out = ["# Literature extraction — all camps",
           "",
           "> **Repaired 2026-09-03** by `experiments/phase_3/src/t0_4_repair_literature.py`.",
           "> The original is preserved at `literature_extraction_all.md`; this file supersedes it.",
           "> Repairs: camp `b`'s header was plain text and is now a heading; three manifest counts",
           "> were wrong; two papers were filed under camp `c` that belong to camps `a` and `b`.",
           "> Ground truth is the directory listing `literature/{a,b,c}/*.md`, 11 files each.",
           ""]
    for c in "abc":
        rows = sorted(body.get(c, []), key=lambda r: r["title"].lower())
        out += [f"# Camp directory name: {c}", "",
                f"Number of `.md` files found in it: {len(truth[c])}", "",
                "List of filenames:"]
        out += [f"- {f}.md" for f in truth[c]]
        out += ["", "Date of extraction: 2026-08-08", ""]
        for r in rows:
            i = next(k for k, t in secs if k + 1 == r["line"])
            lo, hi = span[i]
            out += lines[lo:hi]
        out += [""]
    OUT.write_text("\n".join(out))

    rep = OUT.read_text().split("\n")
    rep_secs = [l[3:].strip() for l in rep if l.startswith("## ")]
    rep_camps = [l for l in rep if re.match(r"^#\s*Camp directory", l)]
    audit["repaired"] = {"path": str(OUT), "n_camp_headers": len(rep_camps),
                         "n_sections": len(rep_secs),
                         "sections_match_dir_total": len(rep_secs) == audit["dir_total"]}
    print(f"\nREPAIRED -> {OUT}")
    print(f"  camp headers {len(rep_camps)} (all now `#` headings)   "
          f"sections {len(rep_secs)} vs {audit['dir_total']} on disk   "
          f"{'OK' if len(rep_secs) == audit['dir_total'] else 'MISMATCH'}")

    json.dump(audit, open(RESULTS, "w"), indent=2)
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()
