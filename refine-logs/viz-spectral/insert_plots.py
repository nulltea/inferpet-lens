#!/usr/bin/env python3
"""Insert the A2/A5 fragments into the report pages, idempotently (skips if the
FIG label is already present). Anchors on a page-unique closing string."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAG = Path(__file__).resolve().parent / "fragments"
HTML = ROOT / "docs/html"

def frag(name): return FRAG.joinpath(name).read_text().rstrip("\n")

# (page, fragment, marker present-check, anchor string to insert BEFORE)
JOBS = [
    ("resid-dp-attacks.html", "a2_dp_attacks.html", "FIG · A2 —",
     '<code>b2_l0_bayes.json</code> (gemma-2-2b, L0).</span></div>\n    </div>\n\n  </section>'),
    ("resid-depth-inversion.html", "a2_depth_inversion.html", "FIG · A2 —",
     'depth buys no privacy.</span></div>\n    </div>\n\n  </section>'),
    ("vec2text.html", "a5_spectrum.html", "FIG · A5 —",
     'read it as a privacy↔utility juxtaposition, not a single joint run.</p></div>\n  </section>'),
    ("probes-registry.html", "a5_spectrum.html", "FIG · A5 —",
     'never as <code>I_G</code>.</p>\n    </div>\n  </section>'),
    ("resid-split.html", "a5_spectrum.html", "FIG · A5 —",
     'infinite <code>I_G</code> and are excluded.</span></div>\n    </div>\n\n  </section>'),
]

for page, fname, marker, anchor in JOBS:
    p = HTML / page
    txt = p.read_text()
    if marker in txt:
        print(f"SKIP {page}: '{marker}' already present")
        continue
    if anchor not in txt:
        print(f"!! ANCHOR NOT FOUND in {page}")
        continue
    block = frag(fname)
    # split the anchor: insert the plot block + blank line right before the closing tag
    head, _, tail = anchor.partition("\n\n  </section>") if "\n\n  </section>" in anchor \
        else anchor.partition("\n  </section>")
    closer = "\n\n  </section>" if "\n\n  </section>" in anchor else "\n  </section>"
    new_anchor = head + "\n" + block + closer + tail
    txt = txt.replace(anchor, new_anchor, 1)
    p.write_text(txt)
    print(f"OK   {page}: inserted {fname}")
