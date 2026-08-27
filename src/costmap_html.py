#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/costmap_html.py -- interactive HTML viewer for the stage-06 cost-map.

Companion to src/route_html.py, same conventions (self-contained inline SVG +
JS, pan/zoom, per-layer toggles) but built around the COMPONENT MAPS instead of
routes:

  * one radio per component layer -- economic balance, air operational safety,
    ground safety, measured traffic, economic value, composite risk, slowness --
    swapped as colour-mapped background images, with an opacity slider;
  * a hover READOUT that reports EVERY layer at once at the cursor, so the
    components can be compared point by point instead of one figure at a time;
  * MODEL NODES as independent toggles: DB bases, DK docks, TN traffic nodes,
    roundabout rings, FLZ emergency landing zones, RA restricted airspace, and
    the raw step-01 model grid;
  * the lane network, optionally coloured by each leg's composite risk (the
    same numbers as network_assessment.csv, in the leg tooltip).

Used by engine_costmap.py; MAKE_HTML=False in the params turns it off.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd

from src.route_html import field_to_datauri

# marker colours shared with the pipeline's matplotlib figures
C_DB, C_DK, C_TN, C_RN = "#c0392b", "#1f6f3f", "#e8710a", "#d000d0"
C_RBT, C_FLZ, C_RA = "#e8352a", "#f39c12", "#8e44ad"


def _pack(a: np.ndarray, vmin: float, vmax: float) -> str:
    """Quantise a field to base64 uint8 for the JS hover readout.
    0 is reserved for 'no value here' (nan / no-fly), real values live in 1..255."""
    a = np.asarray(a, float)
    finite = np.isfinite(a)
    span = max(float(vmax) - float(vmin), 1e-12)
    q = np.clip((a - float(vmin)) / span, 0.0, 1.0)
    q = (1.0 + q * 254.0).round()
    q[~finite] = 0.0
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode()


def _mask_rects(mask: np.ndarray, res: float, H: float) -> str:
    """Run-length merge a Boolean grid into wide SVG rects (fewer elements)."""
    ny, nx = mask.shape
    out = []
    for i in range(ny):
        row = mask[i]
        j = 0
        while j < nx:
            if row[j]:
                j0 = j
                while j < nx and row[j]:
                    j += 1
                out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>'
                           % (j0 * res, H - (i + 1) * res, (j - j0) * res, res))
            else:
                j += 1
    return "".join(out)


def _risk_colour(v: float) -> str:
    """Green -> amber -> red ramp for a 0..1 leg risk (mirrors RdYlGn reversed)."""
    v = float(np.clip(v, 0.0, 1.0))
    stops = [(0.0, (26, 152, 80)), (0.5, (254, 224, 139)), (1.0, (215, 48, 39))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / max(b - a, 1e-9)
            return "#%02x%02x%02x" % tuple(int(round(ca[k] + t * (cb[k] - ca[k])))
                                           for k in range(3))
    return "#d73027"


def render_costmap_html(out_html, layers, extent, res, *, nofly, ra_mask, inside,
                        lanes, legs_assess, nodes, rings, flz_xy, meta,
                        model_dx=None, decimate=3):
    """Write the self-contained stage-06 cost-map viewer.

    layers      : list of (key, title, short, desc, array[ny,nx], cmap, vmin,
                  vmax) -- the component maps; the first is shown on load.
                  `short` labels the compact hover readout, `desc` is the pill's
                  hover tooltip (the Vietnamese name + what the layer means).
    nofly/ra_mask/inside : Boolean grids on the SAME grid as the layers.
    lanes       : stage-05 lane_nodes.csv (leg_id, lane, seq, x, y).
    legs_assess : per-leg assessment rows (leg_id, econ, air, ground, risk,
                  slowness, pair_flow, length_m); drives the leg colouring.
    nodes       : stage-05 network_nodes.csv (net_id, kind, x, y).
    rings       : stage-05 roundabouts.csv, or None.
    flz_xy      : [(x, y, label)] emergency landing zones from step 01.
    model_dx    : step-01 grid spacing (m) for the 'model grid' dot layer.
    """
    ox, oy = float(extent[0]), float(extent[2])
    W = float(extent[1]) - ox
    H = float(extent[3]) - oy
    sx = lambda x: float(x) - ox
    sy = lambda y: H - (float(y) - oy)

    # ---- component layers: colour image + packed values for the readout ----
    imgs, radios, packs, first = [], [], [], None
    for key, title, short, desc, arr, cmap, vmin, vmax in layers:
        uri = field_to_datauri(arr, cmap, vmin, vmax)
        if uri is None:
            continue
        if first is None:
            first = key
        cls = "field" + ("" if key == first else " hidden")
        imgs.append('<g id="f-%s" class="%s"><image x="0" y="0" width="%.1f" '
                    'height="%.1f" href="%s" preserveAspectRatio="none"/></g>'
                    % (key, cls, W, H, uri))
        radios.append('<label class="rad" title="%s"><input type="radio" name="layer" '
                      'value="%s"%s> %s</label>'
                      % (desc, key, " checked" if key == first else "", title))
        packs.append('{k:"%s",t:"%s",lo:%.6g,hi:%.6g,d:"%s"}'
                     % (key, short, vmin, vmax, _pack(arr, vmin, vmax)))
    ny, nx = np.asarray(layers[0][4]).shape

    # ---- base masks ----
    obst_svg = _mask_rects(np.asarray(nofly, bool) & ~np.asarray(ra_mask, bool), res, H)
    ra_svg = _mask_rects(np.asarray(ra_mask, bool), res, H)
    net_svg = _mask_rects(np.asarray(inside, bool), res, H)

    # ---- lanes, coloured by the leg's composite risk ----
    ass = {str(r.leg_id): r for r in legs_assess.itertuples()} if len(legs_assess) else {}
    lane_svg = []
    for (leg_id, lane), g in lanes.groupby(["leg_id", "lane"]):
        xy = g.sort_values("seq")[["x", "y"]].to_numpy(float)
        if len(xy) < 2:
            continue
        if decimate > 1 and len(xy) > 2 * decimate:      # keep the endpoints
            xy = np.vstack([xy[:-1:decimate], xy[-1]])
        pts = " ".join("%.0f,%.0f" % (sx(px), sy(py)) for px, py in xy)
        a = ass.get(str(leg_id))
        col = _risk_colour(a.risk) if a is not None else "#7a8798"
        tip = (("%s (%s)\nrisk %.3f  ->  slowness %.3f\necon %.3f · air %.3f · "
                "ground %.3f\nflow %.0f pair routes · %.0f m")
               % (leg_id, lane, a.risk, a.slowness, a.econ, a.air, a.ground,
                  a.pair_flow, a.length_m)) if a is not None else "%s (%s)" % (leg_id, lane)
        lane_svg.append('<polyline class="lane" stroke="%s" points="%s"><title>%s</title></polyline>'
                        % (col, pts, tip))

    # ---- model nodes, one toggleable group per family ----
    def _lbl(cx, cy, r, text, cls="nlbl"):
        return '<text class="%s" x="%.0f" y="%.0f">%s</text>' % (cls, cx + r + 14, cy + 28, text)

    db, dk, tn, rbt = [], [], [], []
    for r in nodes.itertuples():
        cx, cy = sx(r.x), sy(r.y)
        nid, kind = str(r.net_id), str(r.kind)
        if kind == "objective" and nid.startswith("DB"):
            db.append('<rect class="n db" x="%.0f" y="%.0f" width="60" height="60"><title>%s (base)</title></rect>'
                      % (cx - 30, cy - 30, nid))
            db.append(_lbl(cx, cy, 30, nid))
        elif kind == "objective":
            dk.append('<polygon class="n dk" points="%.0f,%.0f %.0f,%.0f %.0f,%.0f"><title>%s (dock)</title></polygon>'
                      % (cx, cy - 36, cx - 32, cy + 25, cx + 32, cy + 25, nid))
            dk.append(_lbl(cx, cy, 30, nid))
        elif kind == "roundabout":
            rbt.append('<circle class="n rbt" cx="%.0f" cy="%.0f" r="14"><title>%s</title></circle>'
                       % (cx, cy, nid))
        else:                                            # tn_major/minor/backup/ext
            tn.append('<circle class="n tn %s" cx="%.0f" cy="%.0f" r="20"><title>%s (%s)</title></circle>'
                      % (kind, cx, cy, nid, kind))
            tn.append(_lbl(cx, cy, 20, nid, "nlbl small"))

    ring_svg = []
    if rings is not None and len(rings):
        for r in rings.itertuples():
            cx, cy, rr = sx(r.center_x), sy(r.center_y), float(r.radius_m)
            ring_svg.append('<circle class="ring" cx="%.0f" cy="%.0f" r="%.0f">'
                            '<title>%s  r=%.0f m · %s entries · members %s</title></circle>'
                            % (cx, cy, rr, r.rbt_id, rr,
                               getattr(r, "n_entries", "?"), getattr(r, "members", "")))
            ring_svg.append(_lbl(cx, cy, rr, str(r.rbt_id), "nlbl rbtlbl"))

    flz_svg = []
    for x, y, lbl in flz_xy:
        cx, cy = sx(x), sy(y)
        pts = []
        for k in range(10):
            ang = -np.pi / 2 + k * np.pi / 5
            rr = 46 if k % 2 == 0 else 19
            pts.append("%.0f,%.0f" % (cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
        flz_svg.append('<polygon class="n flz" points="%s"><title>%s (emergency landing)</title></polygon>'
                       % (" ".join(pts), lbl))
        flz_svg.append(_lbl(cx, cy, 40, lbl))

    # ---- step-01 model grid as a dot pattern, aligned to the model's own origin ----
    gdx = float(model_dx or res)
    gox = (np.ceil(ox / gdx) * gdx) - ox
    goy = (H - ((np.ceil(oy / gdx) * gdx) - oy)) % gdx

    counts = nodes["kind"].value_counts().to_dict() if len(nodes) else {}
    n_tn = sum(v for k, v in counts.items() if str(k).startswith("tn"))
    meta_line = (
        "%s &nbsp;·&nbsp; grid %d×%d @ %.0f m &nbsp;·&nbsp; weights "
        "econ %.3g / air %.3g / ground %.3g / traffic %.3g &nbsp;·&nbsp; "
        "%d legs, %d roundabouts, %d TN, %d objectives, %d FLZ"
        % (meta.get("subtitle", ""), nx, ny, res,
           meta.get("w_econ", 0), meta.get("w_air", 0),
           meta.get("w_ground", 0), meta.get("w_traffic", 0),
           meta.get("n_legs", 0), 0 if rings is None else len(rings),
           n_tn, counts.get("objective", 0), len(flz_xy)))

    tmpl = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f7f9fc; --ink:#1a2230; --obst:#aeb7c4; --grid:#e6ebf2; }
  * { box-sizing:border-box; }
  /* flex column: the map takes whatever the header + control bars leave, so a
     wrapped control row cannot push the map off the bottom of the window */
  body { margin:0; font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;
         color:var(--ink); background:var(--bg);
         height:100vh; display:flex; flex-direction:column; }
  header, .bar { flex:0 0 auto; }
  header { padding:10px 14px; border-bottom:1px solid var(--grid); }
  h1 { font-size:17px; margin:0 0 3px; }
  .meta { color:#5b6675; font-size:12.5px; }
  .bar { display:flex; flex-wrap:wrap; gap:12px; align-items:center;
         padding:7px 14px; border-bottom:1px solid var(--grid); }
  .bar.layers { background:#fff; }
  .bar b { font-size:11px; letter-spacing:.06em; text-transform:uppercase;
           color:#7a8798; margin-right:2px; }
  label { display:flex; gap:5px; align-items:center; cursor:pointer; }
  .rad { padding:2px 8px; border:1px solid var(--grid); border-radius:14px;
         background:#fbfcfe; }
  .rad:has(input:checked) { background:#1a2230; color:#fff; border-color:#1a2230; }
  .mapbox { position:relative; flex:1 1 auto; min-height:0; width:100%;
            overflow:hidden; background:#fff; touch-action:none; }
  svg { width:100%; height:100%; display:block; cursor:crosshair; }
  svg.grabbing { cursor:grabbing; }
  #obst rect { fill:var(--obst); }
  #ra rect   { fill:rgba(142,68,173,.30); }
  #netmask rect { fill:rgba(20,110,200,.13); }
  .lane { fill:none; stroke-width:11; stroke-linecap:round; stroke-linejoin:round; opacity:.95; }
  #lanes.plain .lane { stroke:#7a8798 !important; opacity:.55; }
  .lane:hover { stroke-width:26; opacity:1; }
  .n { stroke:#fff; stroke-width:4; }
  .n.db { fill:__C_DB__; } .n.dk { fill:__C_DK__; }
  .n.tn { fill:__C_TN__; } .n.tn.tn_minor { fill:#f0a860; }
  .n.tn.tn_backup { fill:__C_RN__; } .n.tn.tn_ext { fill:#8fbf3f; }
  .n.rbt { fill:__C_RBT__; } .n.flz { fill:__C_FLZ__; stroke-width:5; }
  .ring { fill:none; stroke:__C_RBT__; stroke-width:8; opacity:.85; }
  .nlbl { font:bold 74px sans-serif; fill:#141a22; paint-order:stroke;
          stroke:#fff; stroke-width:13px; }
  .nlbl.small { font-size:58px; stroke-width:11px; }
  .nlbl.rbtlbl { fill:#a02018; }
  .frame { fill:none; stroke:#3a4657; stroke-width:9; }
  .hidden { display:none; }
  .readout { position:absolute; left:10px; top:10px; width:236px;
             background:rgba(255,255,255,.95); border:1px solid var(--grid);
             border-radius:8px; padding:9px 11px; font-size:12px;
             box-shadow:0 2px 10px rgba(20,30,45,.10); }
  .readout h4 { margin:0 0 6px; font-size:11px; letter-spacing:.06em;
                text-transform:uppercase; color:#7a8798; }
  .readout .xy { font:12px ui-monospace,SFMono-Regular,Menlo,monospace; color:#5b6675;
                 margin-bottom:6px; }
  .row { display:grid; grid-template-columns:66px 1fr 46px; gap:7px;
         align-items:center; margin:3px 0; line-height:1.1; }
  .row span:first-child { color:#5b6675; white-space:nowrap; overflow:hidden;
                          text-overflow:ellipsis; }
  .row .bar2 { height:7px; background:#eef1f6; border-radius:4px; overflow:hidden; }
  .row .bar2 i { display:block; height:100%; background:#4a6fa5; }
  .row .v { text-align:right; font:12px ui-monospace,Menlo,monospace; }
  .row.hi .bar2 i { background:#d7503a; }
  .legend { position:absolute; right:10px; bottom:10px; background:rgba(255,255,255,.93);
            border:1px solid var(--grid); border-radius:8px; padding:8px 11px; font-size:12px; }
  .legend div { margin:2px 0; }
  .sw { display:inline-block; width:12px; height:12px; border-radius:50%;
        vertical-align:middle; margin-right:7px; border:1px solid #fff; }
  .hint { color:#8a95a5; font-size:11px; }
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
</header>
<div class="bar layers"><b>component map</b>__RADIOS__
  <label class="hint">opacity <input type="range" id="op" min="0" max="100" value="88" style="width:90px"></label>
</div>
<div class="bar">
  <b>model nodes</b>
  <label><input type="checkbox" id="cDB" checked> DB bases</label>
  <label><input type="checkbox" id="cDK" checked> DK docks</label>
  <label><input type="checkbox" id="cTN" checked> TN nodes</label>
  <label><input type="checkbox" id="cRBT" checked> roundabouts</label>
  <label><input type="checkbox" id="cFLZ" checked> FLZ</label>
  <label><input type="checkbox" id="cRA" checked> RA zones</label>
  <label><input type="checkbox" id="cGrid"> model grid</label>
</div>
<div class="bar">
  <b>network</b>
  <label><input type="checkbox" id="cLanes" checked> lanes</label>
  <label><input type="checkbox" id="cRisk" checked> colour lanes by risk</label>
  <label><input type="checkbox" id="cMask"> corridor mask</label>
  <label><input type="checkbox" id="cObst" checked> obstacles</label>
  <label><input type="checkbox" id="cFrame" checked> frame</label>
  <button id="cReset">reset view</button>
  <span class="hint">drag = pan · wheel = zoom · hover = read every layer</span>
</div>
<div class="mapbox">
  <svg id="svg" viewBox="0 0 __W__ __H__" preserveAspectRatio="xMidYMid meet">
    <defs><pattern id="gp" width="__GDX__" height="__GDX__" patternUnits="userSpaceOnUse"
        patternTransform="translate(__GOX__ __GOY__)">
      <circle cx="0" cy="0" r="4" fill="#8592a3"/></pattern></defs>
    <g id="scene">
      <rect x="0" y="0" width="__W__" height="__H__" fill="#fff"/>
      <g id="fields" opacity="0.88">__FIELDS__</g>
      <g id="netmask" class="hidden">__NETMASK__</g>
      <g id="obst">__OBST__</g>
      <g id="ra">__RA__</g>
      <g id="grid" class="hidden"><rect x="0" y="0" width="__W__" height="__H__" fill="url(#gp)"/></g>
      <g id="lanes">__LANES__</g>
      <g id="rings">__RINGS__</g>
      <g id="tn">__TN__</g>
      <g id="rbt">__RBT__</g>
      <g id="flz">__FLZ__</g>
      <g id="dk">__DK__</g>
      <g id="db">__DB__</g>
      <g id="frame"><rect class="frame" x="0" y="0" width="__W__" height="__H__"/></g>
    </g>
  </svg>
  <div class="readout">
    <h4>layers at cursor</h4>
    <div class="xy" id="xy">move over the map</div>
    <div id="rows"></div>
  </div>
  <div class="legend">
    <div><span class="sw" style="background:__C_DB__"></span>DB base</div>
    <div><span class="sw" style="background:__C_DK__"></span>DK dock</div>
    <div><span class="sw" style="background:__C_TN__"></span>TN major <span class="sw" style="background:#f0a860"></span>minor <span class="sw" style="background:__C_RN__"></span>backup <span class="sw" style="background:#8fbf3f"></span>ext</div>
    <div><span class="sw" style="background:__C_RBT__"></span>roundabout ring</div>
    <div><span class="sw" style="background:__C_FLZ__"></span>FLZ landing zone</div>
    <div><span class="sw" style="background:rgba(142,68,173,.6)"></span>RA restricted &nbsp;<span class="sw" style="background:var(--obst)"></span>obstacle</div>
    <div style="margin-top:5px">lane colour = leg risk &nbsp;
      <span style="display:inline-block;width:60px;height:9px;border-radius:3px;
        background:linear-gradient(90deg,#1a9850,#fee08b,#d73027);vertical-align:middle"></span>
      &nbsp;low → high</div>
  </div>
</div>
<script>
(function(){
var NX=__NX__, NY=__NY__, RES=__RES__, X0=__X0__, Y0=__Y0__, H=__H__, W=__W__;
var LAYERS=[__PACKS__];
LAYERS.forEach(function(L){ var b=atob(L.d), a=new Uint8Array(b.length);
  for(var i=0;i<b.length;i++) a[i]=b.charCodeAt(i); L.a=a; });

var svg=document.getElementById('svg'), scene=document.getElementById('scene');
var tx=0, ty=0, s=1;
function apply(){ scene.setAttribute('transform','translate('+tx+' '+ty+') scale('+s+')'); }
function tog(id,el,inv){ var c=document.getElementById(id);
  c.addEventListener('change',function(){ el.classList.toggle('hidden', inv? this.checked : !this.checked); }); }
[['cDB','db'],['cDK','dk'],['cTN','tn'],['cRBT','rbt'],['cFLZ','flz'],
 ['cRA','ra'],['cGrid','grid'],['cLanes','lanes'],['cMask','netmask'],
 ['cObst','obst'],['cFrame','frame']].forEach(function(p){
   tog(p[0], document.getElementById(p[1])); });
document.getElementById('cRBT').addEventListener('change',function(){
  document.getElementById('rings').classList.toggle('hidden', !this.checked); });
document.getElementById('cRisk').addEventListener('change',function(){
  document.getElementById('lanes').classList.toggle('plain', !this.checked); });
document.querySelectorAll('input[name=layer]').forEach(function(r){
  r.addEventListener('change',function(){
    document.querySelectorAll('#fields > g').forEach(function(g){
      g.classList.toggle('hidden', g.id !== 'f-'+r.value); }); }); });
document.getElementById('op').addEventListener('input',function(){
  document.getElementById('fields').setAttribute('opacity', this.value/100); });

// ---- hover readout: every component layer at the cursor ----
var rows=document.getElementById('rows'), xyEl=document.getElementById('xy');
rows.innerHTML = LAYERS.map(function(L){
  return '<div class="row" id="r-'+L.k+'"><span>'+L.t+'</span>'
       + '<span class="bar2"><i style="width:0%"></i></span><span class="v">—</span></div>'; }).join('');
function readout(e){
  var r=svg.getBoundingClientRect(), k=W/r.width;
  var ux=((e.clientX-r.left)*k - tx)/s, uy=((e.clientY-r.top)*k - ty)/s;
  var wx=X0+ux, wy=Y0+(H-uy);
  var ix=Math.floor(ux/RES), iy=Math.floor((H-uy)/RES);
  var inside = ix>=0 && ix<NX && iy>=0 && iy<NY;
  xyEl.textContent = inside ? ('x '+wx.toFixed(0)+' m   y '+wy.toFixed(0)+' m')
                            : 'outside the map';
  LAYERS.forEach(function(L){
    var el=document.getElementById('r-'+L.k), bar=el.querySelector('i'), v=el.querySelector('.v');
    var q = inside ? L.a[iy*NX+ix] : 0;
    if(!q){ bar.style.width='0%'; v.textContent='—'; el.classList.remove('hi'); return; }
    var val = L.lo + (q-1)/254*(L.hi-L.lo), f=(val-L.lo)/Math.max(L.hi-L.lo,1e-9);
    bar.style.width=(f*100).toFixed(1)+'%'; v.textContent=val.toFixed(3);
    el.classList.toggle('hi', f>0.66);
  });
}
svg.addEventListener('pointermove', readout);

// ---- pan / zoom ----
var drag=false, px=0, py=0;
svg.addEventListener('pointerdown',function(e){drag=true;px=e.clientX;py=e.clientY;
  svg.classList.add('grabbing');svg.setPointerCapture(e.pointerId);});
svg.addEventListener('pointermove',function(e){ if(!drag)return;
  var r=svg.getBoundingClientRect(), k=W/r.width;
  tx+=(e.clientX-px)*k; ty+=(e.clientY-py)*k; px=e.clientX; py=e.clientY; apply();});
svg.addEventListener('pointerup',function(){drag=false;svg.classList.remove('grabbing');});
svg.addEventListener('wheel',function(e){ e.preventDefault();
  var r=svg.getBoundingClientRect(), k=W/r.width;
  var mx=(e.clientX-r.left)*k, my=(e.clientY-r.top)*k;
  var f=Math.exp(-e.deltaY*0.0015), ns=Math.min(40,Math.max(0.5,s*f));
  var g=ns/s; tx=mx-(mx-tx)*g; ty=my-(my-ty)*g; s=ns; apply();},{passive:false});
document.getElementById('cReset').addEventListener('click',function(){tx=0;ty=0;s=1;apply();});
})();
</script>
</body></html>
"""
    doc = (tmpl.replace("__TITLE__", str(meta.get("title", "Cost-map")))
               .replace("__META__", meta_line)
               .replace("__RADIOS__", "".join(radios))
               .replace("__FIELDS__", "".join(imgs))
               .replace("__NETMASK__", net_svg)
               .replace("__OBST__", obst_svg)
               .replace("__RA__", ra_svg)
               .replace("__LANES__", "".join(lane_svg))
               .replace("__RINGS__", "".join(ring_svg))
               .replace("__TN__", "".join(tn))
               .replace("__RBT__", "".join(rbt))
               .replace("__FLZ__", "".join(flz_svg))
               .replace("__DK__", "".join(dk))
               .replace("__DB__", "".join(db))
               .replace("__PACKS__", ",".join(packs))
               .replace("__GDX__", "%.3f" % gdx)
               .replace("__GOX__", "%.3f" % gox)
               .replace("__GOY__", "%.3f" % goy)
               .replace("__NX__", str(nx)).replace("__NY__", str(ny))
               .replace("__RES__", "%.6g" % res)
               .replace("__X0__", "%.6g" % ox).replace("__Y0__", "%.6g" % oy)
               .replace("__W__", "%.0f" % W).replace("__H__", "%.0f" % H)
               .replace("__C_DB__", C_DB).replace("__C_DK__", C_DK)
               .replace("__C_TN__", C_TN).replace("__C_RN__", C_RN)
               .replace("__C_RBT__", C_RBT).replace("__C_FLZ__", C_FLZ)
               .replace("__C_RA__", C_RA))
    Path(out_html).write_text(doc, encoding="utf-8")
    return len(doc)
