#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/route_html.py -- shared interactive HTML route-map renderer.

Used by 02_route_plan.py, 04_run_master_corridor_FMM.py and
05_run_master_corridor_theta.py so the HTML output (obstacles, routes,
corridor/buffer bands, DB/DK objectives, the map FRAME, the model's TN/RN
NODES, and the underlying GRID NODES as black dots) stays identical across
planners. Pan/zoom + per-layer toggles, self-contained (inline SVG + JS)."""
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm as _cm, colors as _mcolors


def field_to_datauri(arr, cmap="viridis", vmin=None, vmax=None):
    """Colour-map a 2D field (nan = transparent) to a base64 PNG data-URI, with
    origin='lower' orientation so it drops straight onto the SVG map at y=0..H."""
    a = np.asarray(arr, float)
    finite = np.isfinite(a)
    if not finite.any():
        return None
    vmin = float(np.nanmin(a[finite])) if vmin is None else vmin
    vmax = float(np.nanmax(a[finite])) if vmax is None else vmax
    norm = _mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    rgba = _cm.get_cmap(cmap)(norm(np.where(finite, a, vmin)))
    rgba[~finite] = [0.0, 0.0, 0.0, 0.0]                 # transparent outside/no-fly
    rgba = (rgba[::-1] * 255).astype(np.uint8)           # flip rows -> PNG top = map top
    try:
        from PIL import Image
        buf = BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    except Exception:
        import matplotlib.pyplot as plt
        buf = BytesIO()
        plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_candidate_nodes(xyz_path: Path) -> list:
    """The model's candidate nodes (traffic TN + relief RN) from step-03's
    master_plan_input_nodes.csv, for DISPLAY only (FMM does not route through
    them). Returns [(x, y, label, kind)]; empty for a plain step-01 .xyz with no
    is_candidate column."""
    df = pd.read_csv(xyz_path, sep=None, engine="python")
    if "is_candidate" not in df.columns:
        return []
    is_cand = df["is_candidate"].astype(str).str.lower().isin(["true", "1"])
    out = []
    for r in df[is_cand].itertuples():
        lbl = str(getattr(r, "candidate_id", "") or "")
        ctype = str(getattr(r, "candidate_type", "") or "")
        kind = "TN" if lbl.startswith("TN") else ("RN" if lbl.startswith("RN") else (ctype or "node"))
        out.append((float(r.x), float(r.y), lbl, kind))
    return out




def _corridor_polygon(xy, half_m):
    """Closed left+right offset polygon of a polyline at +-half_m metres."""
    xy = np.asarray(xy, float)
    if len(xy) < 2 or half_m <= 0:
        return None
    d = np.diff(xy, axis=0)
    seglen = np.hypot(d[:, 0], d[:, 1])
    seglen[seglen < 1e-9] = 1e-9
    snx, sny = -d[:, 1] / seglen, d[:, 0] / seglen        # segment unit normals
    vn = np.zeros_like(xy)                                # per-vertex normal (avg)
    vn[:-1, 0] += snx; vn[:-1, 1] += sny
    vn[1:, 0] += snx; vn[1:, 1] += sny
    ln = np.hypot(vn[:, 0], vn[:, 1]); ln[ln < 1e-9] = 1e-9
    vn[:, 0] /= ln; vn[:, 1] /= ln
    left = xy + vn * half_m
    right = xy - vn * half_m
    return np.vstack([left, right[::-1]])



def _route_pair_alt(key: str):
    """Split a route key 'A_to_B#altK' (or 'A_to_B') into (pair, alt_int)."""
    if "#alt" in key:
        pair, a = key.split("#alt", 1)
        try:
            return pair, int(a)
        except ValueError:
            return pair, 0
    return key.split("#", 1)[0], 0



def render_route_html(out_html, routes_xy, obj_xy, nofly, extent, dx,
                      route_width, req_clear, meta, nodes=None, fields=None):
    """Write a self-contained interactive HTML map of the route network.

    Pan (drag) / zoom (wheel); toggle obstacles, the usable corridor band and the
    outer buffer band; filter to one pair; show primary routes only; hover a
    route for its pair / alt / length. No external assets -- inline SVG + JS."""
    ox, oy = extent[0], extent[2]
    W = extent[1] - extent[0]
    H = extent[3] - extent[2]
    ny, nx = nofly.shape

    def sx(x):
        return x - ox

    def sy(y):
        return H - (y - oy)

    # ---- obstacles: per-row run-length merge into wide rects (fewer elements) ----
    rects = []
    for i in range(ny):
        row = nofly[i]
        j = 0
        while j < nx:
            if row[j]:
                j0 = j
                while j < nx and row[j]:
                    j += 1
                rects.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>'
                             % (j0 * dx, H - (i + 1) * dx, (j - j0) * dx, dx))
            else:
                j += 1
    obstacles_svg = "".join(rects)

    # ---- field background layers (cost / risk / conflict as colour-mapped images) ----
    field_layers, field_opts = [], []
    for item in (fields or []):
        fname, farr, fcmap = item[0], item[1], item[2]
        uri = field_to_datauri(farr, fcmap)
        if uri is None:
            continue
        field_layers.append(
            '<g id="field-%s" class="field hidden"><image x="0" y="0" width="%.1f" height="%.1f" '
            'href="%s" preserveAspectRatio="none" opacity="0.9"/></g>' % (fname, W, H, uri))
        field_opts.append('<option value="%s">%s</option>' % (fname, fname))
    field_layers_svg = "".join(field_layers)
    field_opts_svg = "".join(field_opts)

    def poly_pts(poly):
        return " ".join("%.1f,%.1f" % (sx(px), sy(py)) for px, py in poly)

    # ---- routes, corridor bands, buffer bands ----
    buffers, corridors, lines = [], [], []
    pairs_seen = set()
    for key, xy in routes_xy.items():
        pair, alt = _route_pair_alt(key)
        pairs_seen.add(pair)
        xy = np.asarray(xy, float)
        seglen = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        pts = " ".join("%.1f,%.1f" % (sx(px), sy(py)) for px, py in xy)
        cls = "route primary" if alt == 0 else "route alt"
        label = "%s  alt %d  |  %.0f m" % (pair, alt, seglen)
        lines.append('<polyline class="%s" data-pair="%s" data-alt="%d" '
                     'points="%s"><title>%s</title></polyline>'
                     % (cls, pair, alt, pts, label))
        bpoly = _corridor_polygon(xy, req_clear)
        if bpoly is not None:
            buffers.append('<polygon class="buffer" data-pair="%s" points="%s"/>'
                           % (pair, poly_pts(bpoly)))
        cpoly = _corridor_polygon(xy, 0.5 * route_width)
        if cpoly is not None:
            corridors.append('<polygon class="corridor" data-pair="%s" points="%s"/>'
                             % (pair, poly_pts(cpoly)))

    # ---- objectives ----
    objs = []
    for nid, (x, y) in obj_xy.items():
        cx, cy = sx(x), sy(y)
        is_db = str(nid).startswith("DB")
        if is_db:
            objs.append('<rect class="obj db" x="%.1f" y="%.1f" width="34" height="34"/>'
                        % (cx - 17, cy - 17))
        else:
            objs.append('<polygon class="obj dk" points="%.1f,%.1f %.1f,%.1f %.1f,%.1f"/>'
                        % (cx, cy - 20, cx - 18, cy + 14, cx + 18, cy + 14))
        objs.append('<text class="objlbl" x="%.1f" y="%.1f">%s</text>'
                    % (cx + 20, cy - 8, nid))
    objs_svg = "".join(objs)

    # ---- model frame (map boundary) + model nodes (TN/RN candidates) ----
    # nodes items: (x, y, label, kind) or (x, y, label, kind, used_bool). When a
    # `used` flag is given, USED nodes are drawn solid and UNUSED ones hollow/faded,
    # so you can see how many of the model's nodes the route network actually uses.
    frame_svg = '<rect class="frame" x="0" y="0" width="%.1f" height="%.1f"/>' % (W, H)
    mnodes = []
    n_used = 0
    have_used = any(len(n) > 4 for n in (nodes or []))
    for item in (nodes or []):
        x, y, lbl, kind = item[0], item[1], item[2], item[3]
        used = bool(item[4]) if len(item) > 4 else True
        if used:
            n_used += 1
        cx, cy = sx(x), sy(y)
        cls = ("mnode tn" if kind == "TN" else "mnode rn") + ("" if used else " unused")
        mnodes.append('<circle class="%s" cx="%.1f" cy="%.1f" r="16"><title>%s%s</title></circle>'
                      % (cls, cx, cy, lbl, "" if used else " (unused)"))
        mnodes.append('<text class="mnodelbl" x="%.1f" y="%.1f">%s</text>' % (cx + 20, cy + 6, lbl))
    nodes_svg = "".join(mnodes)
    n_tn = sum(1 for n in (nodes or []) if n[3] == "TN")
    n_rn = sum(1 for n in (nodes or []) if n[3] == "RN")

    opts = "".join('<option value="%s">%s</option>' % (p, p)
                   for p in sorted(pairs_seen))
    meta_line = (
        "planner %s &nbsp;·&nbsp; K=%s &nbsp;·&nbsp; %d routes / %d pairs "
        "&nbsp;·&nbsp; width %.0f m + buffer %.0f m (band ±%.0f m) "
        "&nbsp;·&nbsp; weights t%.2g/r%.2g/c%.2g"
        % (meta.get("planner", "?"), meta.get("diversify_k", "?"),
           meta.get("n_routes", 0), meta.get("n_pairs", 0),
           route_width, req_clear - 0.5 * route_width, req_clear,
           meta.get("w_time", 0), meta.get("w_risk", 0), meta.get("w_conflict", 0)))
    if nodes:
        meta_line += "  &nbsp;·&nbsp; model nodes: %d TN + %d RN" % (n_tn, n_rn)
        if have_used:
            meta_line += "  &nbsp;·&nbsp; <b>%d/%d nodes USED</b> by routes" % (n_used, n_tn + n_rn)

    tmpl = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f7f9fc; --ink:#1a2230; --obst:#aeb7c4; --grid:#e6ebf2; }
  * { box-sizing:border-box; }
  body { margin:0; font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;
         color:var(--ink); background:var(--bg); }
  header { padding:10px 14px; border-bottom:1px solid var(--grid); }
  h1 { font-size:15px; margin:0 0 3px; }
  .meta { color:#5b6675; font-size:12px; }
  .controls { display:flex; flex-wrap:wrap; gap:14px; align-items:center;
              padding:8px 14px; border-bottom:1px solid var(--grid); }
  .controls label { display:flex; gap:5px; align-items:center; cursor:pointer; }
  .mapbox { position:relative; width:100%; height:calc(100vh - 118px);
            overflow:hidden; background:#fff; touch-action:none; }
  svg { width:100%; height:100%; display:block; cursor:grab; }
  svg.grabbing { cursor:grabbing; }
  #obst rect { fill:var(--obst); }
  .buffer  { fill:rgba(16,40,220,.09); stroke:none; }
  .corridor{ fill:rgba(16,40,220,.20); stroke:none; }
  .route   { fill:none; stroke-linejoin:round; stroke-linecap:round; }
  .route.primary { stroke:#1030e0; stroke-width:2.2; opacity:.95; }
  .route.alt     { stroke:#14b7c2; stroke-width:1.3; opacity:.6; }
  .route:hover   { stroke:#ff7a00; stroke-width:3.4; opacity:1; }
  .obj.db { fill:#c0392b; stroke:#222; stroke-width:1.5; }
  .obj.dk { fill:#1f6f3f; stroke:#222; stroke-width:1.5; }
  .objlbl { font:bold 15px sans-serif; fill:#111; paint-order:stroke;
            stroke:#fff; stroke-width:3px; }
  .frame { fill:none; stroke:#3a4657; stroke-width:8; }
  .mnode { stroke:#fff; stroke-width:2.5; }
  .mnode.tn { fill:#e8710a; }
  .mnode.rn { fill:#d000d0; }
  .mnode.unused { fill:#fff; stroke-width:3.5; opacity:.9; }
  .mnode.tn.unused { stroke:#e8710a; }
  .mnode.rn.unused { stroke:#d000d0; }
  .mnodelbl { font:bold 22px sans-serif; fill:#222; paint-order:stroke;
              stroke:#fff; stroke-width:4px; }
  .hidden { display:none; }
  .legend { position:absolute; right:10px; bottom:10px; background:rgba(255,255,255,.9);
            border:1px solid var(--grid); border-radius:6px; padding:8px 10px; font-size:12px; }
  .legend i { display:inline-block; width:22px; height:0; vertical-align:middle;
              border-top-width:3px; border-top-style:solid; margin-right:6px; }
  .hint { color:#8a95a5; font-size:11px; }
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
</header>
<div class="controls">
  <label><input type="checkbox" id="cObst" checked> obstacles</label>
  <label><input type="checkbox" id="cFrame" checked> frame</label>
  <label><input type="checkbox" id="cGrid"> grid nodes</label>
  <label><input type="checkbox" id="cNodes" checked> model nodes</label>
  <label><input type="checkbox" id="cCorr"> corridor band</label>
  <label><input type="checkbox" id="cBuf"> buffer band</label>
  <label><input type="checkbox" id="cPrim"> primary routes only</label>
  <label>pair:
    <select id="cPair"><option value="all">all</option>__OPTS__</select>
  </label>
  <label>field bg:
    <select id="cField"><option value="none">none</option>__FIELDOPTS__</select>
  </label>
  <button id="cReset">reset view</button>
  <span class="hint">drag = pan · wheel = zoom</span>
</div>
<div class="mapbox">
  <svg id="svg" viewBox="0 0 __W__ __H__" preserveAspectRatio="xMidYMid meet">
    <defs><pattern id="gridpat" width="__DX__" height="__DX__" patternUnits="userSpaceOnUse">
      <circle cx="0" cy="0" r="3.5" fill="#111"/></pattern></defs>
    <g id="scene">
      <rect x="0" y="0" width="__W__" height="__H__" fill="#fff"/>
      <g id="fields">__FIELDS__</g>
      <g id="obst">__OBST__</g>
      <g id="grid" class="hidden"><rect x="0" y="0" width="__W__" height="__H__" fill="url(#gridpat)"/></g>
      <g id="bufs" class="hidden">__BUFS__</g>
      <g id="corr" class="hidden">__CORR__</g>
      <g id="routes">__LINES__</g>
      <g id="nodes">__NODES__</g>
      <g id="objs">__OBJS__</g>
      <g id="frame">__FRAME__</g>
    </g>
  </svg>
  <div class="legend">
    <div><i style="border-color:#1030e0"></i>primary route</div>
    <div><i style="border-color:#14b7c2"></i>alternate route</div>
    <div><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#111;vertical-align:middle;margin-right:9px;margin-left:2px"></span>grid node</div>
    <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#e8710a;vertical-align:middle;margin-right:7px"></span>traffic node (TN)</div>
    <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#d000d0;vertical-align:middle;margin-right:7px"></span>relief node (RN)</div>
    <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#fff;border:2px solid #888;vertical-align:middle;margin-right:8px"></span>node NOT used (hollow)</div>
    <div><span style="display:inline-block;width:16px;height:12px;background:var(--obst);vertical-align:middle;margin-right:6px"></span>no-fly / obstacle</div>
  </div>
</div>
<script>
(function(){
  var svg=document.getElementById('svg'), scene=document.getElementById('scene');
  var tx=0, ty=0, s=1;
  function apply(){ scene.setAttribute('transform','translate('+tx+' '+ty+') scale('+s+')'); }
  // toggles
  function tog(id, node){ document.getElementById(id).addEventListener('change',function(){
      node.classList.toggle('hidden', !this.checked); }); }
  tog('cObst', document.getElementById('obst'));
  tog('cGrid', document.getElementById('grid'));
  document.getElementById('cField').addEventListener('change', function(){
      var v = this.value;
      document.querySelectorAll('#fields .field').forEach(function(g){
          g.classList.toggle('hidden', g.id !== 'field-' + v); }); });
  tog('cFrame', document.getElementById('frame'));
  tog('cNodes', document.getElementById('nodes'));
  document.getElementById('cCorr').addEventListener('change',function(){
      document.getElementById('corr').classList.toggle('hidden', !this.checked); });
  document.getElementById('cBuf').addEventListener('change',function(){
      document.getElementById('bufs').classList.toggle('hidden', !this.checked); });
  function applyFilters(){
    var only=document.getElementById('cPrim').checked;
    var pair=document.getElementById('cPair').value;
    var all=document.querySelectorAll('#routes .route, #corr .corridor, #bufs .buffer');
    all.forEach(function(el){
      var isPrim = el.classList.contains('primary');
      var okPrim = !only || el.classList.contains('buffer') || el.classList.contains('corridor') || isPrim;
      var okPair = (pair==='all') || (el.getAttribute('data-pair')===pair);
      el.classList.toggle('hidden', !(okPrim && okPair));
    });
  }
  document.getElementById('cPrim').addEventListener('change', applyFilters);
  document.getElementById('cPair').addEventListener('change', applyFilters);
  // pan
  var drag=false, px=0, py=0;
  svg.addEventListener('pointerdown',function(e){drag=true;px=e.clientX;py=e.clientY;
      svg.classList.add('grabbing');svg.setPointerCapture(e.pointerId);});
  svg.addEventListener('pointermove',function(e){ if(!drag)return;
      var r=svg.getBoundingClientRect(); var k=__W__/r.width;
      tx+=(e.clientX-px)*k; ty+=(e.clientY-py)*k; px=e.clientX; py=e.clientY; apply();});
  svg.addEventListener('pointerup',function(e){drag=false;svg.classList.remove('grabbing');});
  // zoom to cursor
  svg.addEventListener('wheel',function(e){ e.preventDefault();
      var r=svg.getBoundingClientRect(); var k=__W__/r.width;
      var mx=(e.clientX-r.left)*k, my=(e.clientY-r.top)*k;
      var f=Math.exp(-e.deltaY*0.0015); var ns=Math.min(40,Math.max(0.5,s*f));
      var g=ns/s; tx=mx-(mx-tx)*g; ty=my-(my-ty)*g; s=ns; apply();},{passive:false});
  document.getElementById('cReset').addEventListener('click',function(){tx=0;ty=0;s=1;apply();});
})();
</script>
</body></html>
"""
    doc = (tmpl.replace("__TITLE__", str(meta.get("title", "Route network")))
               .replace("__META__", meta_line)
               .replace("__OPTS__", opts)
               .replace("__DX__", "%.3f" % dx)
               .replace("__FIELDS__", field_layers_svg)
               .replace("__FIELDOPTS__", field_opts_svg)
               .replace("__W__", "%.0f" % W)
               .replace("__H__", "%.0f" % H)
               .replace("__OBST__", obstacles_svg)
               .replace("__BUFS__", "".join(buffers))
               .replace("__CORR__", "".join(corridors))
               .replace("__LINES__", "".join(lines))
               .replace("__NODES__", nodes_svg)
               .replace("__FRAME__", frame_svg)
               .replace("__OBJS__", objs_svg))
    Path(out_html).write_text(doc, encoding="utf-8")


