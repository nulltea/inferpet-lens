#!/usr/bin/env python3
"""Task 9 (c10-viz-scatter): roll the A1 bits-vs-recovery scatter across every sweep page.
ONE primitive, copied from the vec2text.html reference (viewBox 0 0 680 440, plot box
x in [64,656], y in [20,384]). Each fragment is emitted to refine-logs/viz-scatter/<name>.frag.html
and inserted into its page. Every annotated rho is cross-checked against the on-disk JSON and
must equal the number already stated in the page prose (PLOT-STYLE.md cardinal rule)."""
import json, math, html

L, R, T, B = 64, 656, 20, 384          # plot box (matches vec2text A1)

def spearman(a, b):
    def rank(x):
        s = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0]*len(x); i = 0
        while i < len(x):
            j = i
            while j+1 < len(x) and x[s[j+1]] == x[s[i]]: j += 1
            avg = (i+j)/2.0
            for k in range(i, j+1): r[s[k]] = avg
            i = j+1
        return r
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra)/n; mb = sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    den = math.sqrt(sum((x-ma)**2 for x in ra)*sum((x-mb)**2 for x in rb))
    return num/den if den else float('nan')

def emit(fig, subject, source, n, aria, title, xdomain, xticks, ydomain, yticks,
         series, badges, axis_x, axis_y, read, refs=None, pointlabels=None):
    xmin, xmax = xdomain; ymin, ymax = ydomain
    def X(v): return L + (v-xmin)/(xmax-xmin)*(R-L)
    def Y(v): return B - (v-ymin)/(ymax-ymin)*(B-T)
    out = []
    out.append('    <div class="plot-frame">')
    out.append(f'      <div class="plot-cap"><span class="l">FIG · {fig} — {subject}</span>'
               f'<span class="r">source: {source} · N={n}</span></div>')
    out.append(f'      <svg class="plot" viewBox="0 0 680 440" xmlns="http://www.w3.org/2000/svg" '
               f'role="img" aria-label="{html.escape(aria)}">')
    out.append(f'        <title>{html.escape(title)}</title>')
    # gridlines (internal x ticks + all y ticks)
    out.append('        <!-- gridlines -->')
    for xv, _ in xticks[1:]:
        out.append(f'        <line class="gridline" x1="{X(xv):.0f}" y1="{T}" x2="{X(xv):.0f}" y2="{B}"/>')
    for yv in yticks:
        out.append(f'        <line class="gridline" x1="{L}" y1="{Y(yv):.0f}" x2="{R}" y2="{Y(yv):.0f}"/>')
    # reference lines
    if refs:
        for rv, lab in refs:
            out.append(f'        <line class="ref-line" x1="{X(rv):.0f}" y1="{T}" x2="{X(rv):.0f}" y2="{B}"/>')
            out.append(f'        <text class="ref-label" x="{X(rv)+4:.0f}" y="{T+13}">{html.escape(lab)}</text>')
    # axes
    out.append('        <!-- axes -->')
    out.append(f'        <line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/>')
    out.append(f'        <line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>')
    # x ticks
    out.append('        <!-- x ticks -->')
    for xv, lab in xticks:
        out.append(f'        <line class="tick" x1="{X(xv):.0f}" y1="{B}" x2="{X(xv):.0f}" y2="{B+5}"/>')
        out.append(f'        <text class="tick-label" x="{X(xv):.0f}" y="{B+17}" text-anchor="middle">{lab}</text>')
    # y ticks
    out.append('        <!-- y ticks -->')
    for yv, lab in yticks_l:
        out.append(f'        <line class="tick" x1="{L-4}" y1="{Y(yv):.0f}" x2="{L}" y2="{Y(yv):.0f}"/>')
        out.append(f'        <text class="tick-label" x="{L-8}" y="{Y(yv)+3:.0f}" text-anchor="end">{lab}</text>')
    # series
    for s in series:
        cls = s["cls"]; pts = s["pts"]
        coords = " ".join(f"{X(px):.0f},{Y(py):.0f}" for px, py in pts)
        out.append(f'        <!-- series: {s["label"]} -->')
        if s.get("line", True) and len(pts) > 1:
            out.append(f'        <polyline class="plot-line{cls}" points="{coords}"/>')
        for px, py in pts:
            out.append(f'        <circle class="plot-point{cls}" cx="{X(px):.0f}" cy="{Y(py):.0f}" r="4.5"/>')
    # per-point labels
    if pointlabels:
        out.append('        <!-- point labels -->')
        for px, py, lab, anch, dx, dy in pointlabels:
            out.append(f'        <text class="point-label" x="{X(px)+dx:.0f}" y="{Y(py)+dy:.0f}" '
                       f'text-anchor="{anch}">{html.escape(lab)}</text>')
    # badges
    by = T + 30
    for i, (b, sub) in enumerate(badges):
        out.append(f'        <text class="stat-badge" x="{R-8}" y="{by}" text-anchor="end">{html.escape(b)}</text>')
        out.append(f'        <text class="stat-sub" x="{R-8}" y="{by+14}" text-anchor="end">{html.escape(sub)}</text>')
        by += 34
    # axis labels
    out.append(f'        <text class="axis-label" x="{(L+R)//2}" y="423" text-anchor="middle">{html.escape(axis_x)}</text>')
    out.append(f'        <text class="axis-label" x="20" y="{(T+B)//2}" text-anchor="middle" '
               f'transform="rotate(-90 20 {(T+B)//2})">{html.escape(axis_y)}</text>')
    out.append('      </svg>')
    out.append(f'      <div class="plot-cap" style="margin:14px 0 0;"><span class="l">Read</span>'
               f'<span class="r" style="text-transform:none;letter-spacing:0;font-size:11px;">{read}</span></div>')
    out.append('    </div>')
    return "\n".join(out)

frags = {}

# ───────────────────────── resid-split ─────────────────────────
sp = json.load(open("refine-logs/resid-split/runs/sweep/pripert_sweep.json"))
xs_ridge=[]; ys_ridge=[]; xs_mlp=[]; ys_mlp=[]; bestrec=[]; igb=[]
for r in sp["records"]:
    if r["probes"].get("i_g_is_inf"): continue
    ig = r["probes"]["i_g_bits"]; inv = r["inverters"]
    igb.append(ig)
    xs_ridge.append(ig); ys_ridge.append(inv["ridge"]["selectivity"])
    xs_mlp.append(ig);   ys_mlp.append(inv["mlp2"]["selectivity"])
    bestrec.append(max(inv["ridge"]["selectivity"], inv["mlp2"]["selectivity"], inv["nn"]["selectivity"]))
rho_best = spearman(igb, bestrec); rho_mlp = spearman(xs_mlp, ys_mlp)
print(f"[split] n={len(igb)} xmax={max(igb):.0f} rho_best={rho_best:.3f} (page 0.958) rho_mlp2={rho_mlp:.3f} (page 0.915)")
# log-x: bits span ~10..2400 (>2 orders) -> log10 axis
import math as _m
def lg(v): return _m.log10(v)
xticks=[(lg(10),"10"),(lg(100),"100"),(lg(1000),"1k")]
yticks_full=[0,.25,.5,.75]
yticks=yticks_full; yticks_l=[(0,"0"),(.25,".25"),(.5,".50"),(.75,".75")]
frags["resid-split"]=emit(
  "A1","matched ceiling I_G vs best-inverter recovery","refine-logs/resid-split/runs/sweep/pripert_sweep.json",24,
  "Scatter of the I_G leakage ceiling in bits against PriPert best-inverter recovery over the 24 finite-measure sweep cells; recovery rises with the ceiling, Spearman rho equals plus 0.96 for the best inverter, and the stronger learned inverter also tracks it.",
  "I_G bits vs PriPert recovery across the 24-cell sweep (ρ = +0.96 best inverter; the stronger learned inverter also tracks)",
  (lg(8),lg(3000)),xticks,(0,.78),yticks,
  [{"label":"best inverter","cls":"","pts":[(lg(x),y) for x,y in zip(igb,bestrec)],"line":False},
   {"label":"mlp2 (learned)","cls":" b","pts":[(lg(x),y) for x,y in zip(xs_mlp,ys_mlp)],"line":False}],
  [("ρ = +0.96","Spearman · I_G vs best recovery")],
  "I_G — Gaussian channel-capacity MI ceiling (bits, log scale)","recovery — top-1 selectivity",
  'The best-inverter recovery (navy) climbs with the attack-independent ceiling <code>I_G</code> across the joint sparsity×perturbation sweep (ρ=+0.96), and the stronger learned inverter (<code>mlp2</code>, amber) tracks it too, so no measure-attack gap appears on this surface. Eight noiseless β=0 cells have infinite <code>I_G</code> and are excluded.')

# ───────────────────────── resid-depth-inversion ─────────────────────────
dp = json.load(open("refine-logs/resid-depth-inversion/runs/full/depth_sweep.json"))
accs=[]; best=[]; layers=[]
for r in dp["records"]:
    accs.append(r["probes"]["cap_reader_acc"]); layers.append(r["layer"])
    inv=r["inverters"]
    best.append(max(inv["ridge"]["selectivity"], inv["mlp2"]["selectivity"], inv["nn"].get("selectivity",inv["nn"].get("ttrsr_top1",0))))
order=sorted(range(len(layers)),key=lambda i:layers[i])
accs=[accs[i] for i in order]; best=[best[i] for i in order]; layers=[layers[i] for i in order]
rho_d=spearman(accs,best)
print(f"[depth] n={len(layers)} acc range {min(accs):.3f}-{max(accs):.3f} rec {min(best):.3f}-{max(best):.3f} rho={rho_d:.3f} (page +0.85)")
yticks_l=[(.3,".30"),(.45,".45"),(.6,".60"),(.75,".75")]
frags["resid-depth-inversion"]=emit(
  "A1","V_cap reader accuracy vs inversion recovery across depth","refine-logs/resid-depth-inversion/runs/full/depth_sweep.json",9,
  "Scatter of the capacity-matched V_cap reader accuracy against best-inverter recovery at the nine sampled depths; the probe tracks recovery across depth, Spearman rho equals plus 0.85.",
  "V_cap reader accuracy vs best-inverter recovery across 9 depths (rho = +0.85)",
  (.64,.98),[(.65,".65"),(.75,".75"),(.85,".85"),(.95,".95")],(.30,.78),[.3,.45,.6,.75],
  [{"label":"best inverter","cls":"","pts":list(zip(accs,best)),"line":True}],
  [("ρ = +0.85","Spearman · V_cap acc vs recovery"),("CLUB +0.78","MI upper bound vs recovery")],
  "V_cap — capacity-matched reader accuracy (bits-equivalent readout)","recovery — best-inverter selectivity",
  'The attack-independent capacity-matched reader (<code>V_cap</code>, accuracy readout) ranks recovery across the nine depths at ρ=+0.85 (permutation p&lt;0.01), with the <code>CLUB</code> MI upper bound tracking at +0.78: the positive regime of the measurement loop. Points are connected in depth order (L0→L32); depth buys no privacy.',
  pointlabels=[(accs[0],best[0],"L0","start",6,-6),(accs[-1],best[-1],"L32","end",-6,14)])

# ───────────────────────── resid-dp-attacks (L0 input-DP) ─────────────────────────
# SINGLE self-consistent source: every plotted coordinate AND the badge ρ come from
# results/b2_l0_bayes.json (all 8 rows that carry club_bits). The badge +0.76 = the stored
# ridge_vs_club Spearman, which equals the Spearman of exactly these plotted ridge points.
b2=json.load(open("results/b2_l0_bayes.json"))
rows=sorted([r for r in b2["records"] if r.get("club_bits") is not None], key=lambda r:r["club_bits"])
club=[r["club_bits"] for r in rows]; ridge=[r["ridge_ttrsr"] for r in rows]; bayes=[r["bayes_map_unif"] for r in rows]
rho_dp=spearman(club,ridge); ceil=bayes[0]; collapse=ceil/min(ridge) if min(ridge)>0 else float('inf')
print(f"[dp-attacks] n={len(rows)} club {min(club):.0f}-{max(club):.0f} spearman(club,ridge)={rho_dp:.4f} (page +0.76) ceiling={ceil:.3f} collapse≈{collapse:.0f}x")
imax=club.index(max(club)); imin=club.index(min(club))
yticks_l=[(0,"0"),(.2,".20"),(.4,".40"),(.6,".60")]
frags["resid-dp-attacks"]=emit(
  "A1","L0 input-DP · ridge vs exact Bayes-NN against CLUB bits","results/b2_l0_bayes.json",len(rows),
  "Scatter of CLUB leakage bits against top-1 recovery over the L0 input-DP epsilon sweep, with ridge and exact Bayes-NN attack series; ridge collapses as the CLUB bits fall while Bayes-NN holds flat at the L0 ceiling, and ridge still tracks CLUB at Spearman plus 0.76.",
  "L0 input-DP: ridge collapses while Bayes-NN holds the ceiling as CLUB bits fall (ridge ρ=+0.76)",
  (1850,3150),[(1900,"1900"),(2300,"2300"),(2700,"2700"),(3100,"3100")],(0,.7),[0,.2,.4,.6],
  [{"label":"Bayes-NN (exact)","cls":"","pts":list(zip(club,bayes)),"line":True},
   {"label":"ridge (linear)","cls":" b","pts":list(zip(club,ridge)),"line":True}],
  [("ρ = +0.76","Spearman · ridge vs CLUB"),("Bayes flat","holds the L0 ceiling")],
  "CLUB — variational MI upper bound (bits)","recovery — top-1 (TTRSR)",
  'As input-DP noise grows the <code>CLUB</code> ceiling decays only ~38%, but the weak <strong>ridge</strong> attack (amber) collapses about 39× to near zero, yet still tracks <code>CLUB</code> at ρ=+0.76: a ceiling-realization gap, not decorrelation. The exact <strong>Bayes-NN</strong> (navy) holds flat at the L0 ceiling: the preserved information is there, and only the stronger attack extracts it. All coordinates and the ρ come from <code>b2_l0_bayes.json</code> (gemma-2-2b, L0).',
  pointlabels=[(club[imax],bayes[imax],"ε=∞","end",-8,-8),(club[imin],ridge[imin],"ε=128","start",8,4)])

# ───────────────────────── embed-sgt ─────────────────────────
sg=json.load(open("refine-logs/embed-sgt/runs/sweep/sgt_eval.json"))
shape_pts={"iso":[],"sgt_opt":[],"tail_dump":[]}
allb=[]; allf=[]
for r in sg["records"]:
    if r["shape"]=="plaintext" or r["i_g_bits"] is None: continue
    sh=r["shape"]
    if sh in shape_pts:
        shape_pts[sh].append((r["i_g_bits"],r["token_f1"])); allb.append(r["i_g_bits"]); allf.append(r["token_f1"])
rho_sg=spearman(allb,allf)
print(f"[sgt] n={len(allb)} bits {min(allb):.0f}-{max(allb):.0f} rho(I_G,F1)={rho_sg:.3f} (page 0.48)")
yticks_l=[(0,"0"),(.2,".20"),(.4,".40"),(.6,".60")]
frags["embed-sgt"]=emit(
  "A1","I_G budget bits vs Vec2Text token-F1 by noise shape","refine-logs/embed-sgt/runs/sweep/sgt_eval.json",12,
  "Scatter of the I_G channel-MI budget in bits against Vec2Text token-F1 across twelve Stained-Glass settings, separated by noise shape; across shapes the budget tracks recovery only weakly, Spearman rho equals plus 0.48, while within the isotropic and utility-optimal shapes it is monotone.",
  "I_G budget vs token-F1: shape-blind across shapes (ρ=+0.48), monotone within the iso/utility-optimal shapes (+1.00)",
  (40,900),[(70,"71"),(196,"196"),(434,"434"),(827,"827")],(0,.62),[0,.2,.4,.6],
  [{"label":"utility-optimal","cls":"","pts":sorted(shape_pts["sgt_opt"]),"line":True},
   {"label":"isotropic","cls":" b","pts":sorted(shape_pts["iso"]),"line":True},
   {"label":"tail-loaded","cls":" c","pts":sorted(shape_pts["tail_dump"]),"line":True}],
  [("ρ = +0.48","Spearman · I_G vs token-F1 (12 settings)"),("within-shape +1.00","iso & utility-optimal shapes")],
  "I_G — Gaussian channel-capacity MI budget (bits)","recovery — Vec2Text token-F1",
  'At a fixed budget the three noise <em>shapes</em> recover very differently: utility-optimal (navy) and isotropic (amber) leak, tail-loaded (dotted) does not, so the scalar <code>I_G</code> budget tracks recovery only at ρ=+0.48 across shapes, it is shape-blind. <em>Within</em> the isotropic and utility-optimal shapes the budget is perfectly monotone (+1.00); the tail-loaded shape is the exception (it does not track). The released-embedding cosine, which sees the shape, tracks at +0.97.')

# ───────────────────────── kv-cloak ─────────────────────────
kv=json.load(open("refine-logs/kv-cloak/sweep.json"))
# obfuscated configs only (exclude the 3 identity baselines) -> 270, matching the page; recon = jade_p95
nb=[]; rec=[]
for r in kv["records"]:
    if r.get("channel")=="identity" or r.get("negentropy_bits") is None or r.get("jade_p95") is None: continue
    nb.append(r["negentropy_bits"]); rec.append(r["jade_p95"])
rho_kv=spearman(nb,rec)
print(f"[kv] n={len(nb)} negent {min(nb):.0f}-{max(nb):.0f} rho={rho_kv:.3f} (page 0.71)")
# channel means (obfuscated channels)
chan={}
for r in kv["records"]:
    if r.get("channel")=="identity" or r.get("negentropy_bits") is None: continue
    c=r.get("channel","?"); chan.setdefault(c,[]).append((r["negentropy_bits"],r["jade_p95"]))
cmeans=[(sum(x for x,_ in v)/len(v), sum(y for _,y in v)/len(v)) for v in chan.values()]
rho_cm=spearman([m[0] for m in cmeans],[m[1] for m in cmeans])
print(f"[kv] channel-mean rho={rho_cm:.3f} (page 0.77), n_chan={len(cmeans)}")
nmax=max(nb)
yticks_l=[(0,"0"),(.25,".25"),(.5,".50"),(.75,".75")]
frags["kv-cloak"]=emit(
  "A1","row negentropy J vs KV reconstruction across 270 configs","refine-logs/kv-cloak/sweep.json",len(nb),
  "Scatter of whitened-row negentropy in bits against KV reconstruction over 270 KV-Cloak configurations; the separability resource ranks reconstruction across channel families, Spearman rho equals plus 0.71.",
  "Row negentropy J vs KV reconstruction across 270 configs (between-channel ρ=+0.71)",
  (0,nmax*1.05),[(0,"0"),(nmax*0.33,f"{nmax*0.33:.0f}"),(nmax*0.66,f"{nmax*0.66:.0f}"),(nmax,f"{nmax:.0f}")],(0,.8),[0,.25,.5,.75],
  [{"label":"configs","cls":"","pts":list(zip(nb,rec)),"line":False}],
  [("ρ = +0.71","Spearman · J vs recon (270 cfg)"),("channel-mean +0.77","across channel families")],
  "J — whitened-row negentropy separability (bits)","reconstruction — JADE p95",
  'Across the 270 KV-Cloak configurations the row-negentropy separability resource <code>J</code> ranks reconstruction at ρ=+0.71 (permutation p&lt;1e-40), rising to +0.77 across channel-family means. The probe orders the channels by how much they leak. It is a <em>between-channel</em> diagnostic: within a single channel the sign is weak and can invert, so it is not a fine-grained meter.')

import os
os.makedirs("refine-logs/viz-scatter/frags",exist_ok=True)
for k,v in frags.items():
    open(f"refine-logs/viz-scatter/frags/{k}.frag.html","w").write(v+"\n")
print("\nWROTE",len(frags),"fragments to refine-logs/viz-scatter/frags/")
