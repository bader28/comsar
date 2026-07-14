"""comsar/viz.py -- Interactive visualisation helpers for Jupyter notebooks.

Currently provides :func:`timbre_player`, an audio player that draws the
waveform in light grey with the extracted feature tracks overlaid in colour,
plus a play button and a cursor that follows the playback position. When a
table of partials is passed, an additional "partial-gram" panel shows the
detected partial frequencies over time (grey level = amplitude).

The widget is fully self-contained HTML/JavaScript (the audio is embedded as a
data URI), so it keeps working when the notebook is exported to HTML.
"""
import base64
import io
import json
from html import escape

import numpy as np
import soundfile as sf


# Runs inside a sandboxed <iframe srcdoc> so the JavaScript also works in
# JupyterLab (which does not execute <script> tags in plain HTML output).
_PLAYER_DOC = r"""<!doctype html><html><head><meta charset="utf-8"><style>
  :root{--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--wave:#d6d6d6;}
  body{margin:0;font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;
       color:var(--ink);background:var(--surface);}
  #wrap{position:relative;}
  canvas{display:block;width:100%;}
  #cursor{position:absolute;top:0;width:2px;background:#0b0b0b;opacity:.55;
          pointer-events:none;left:0;}
  #bar{display:flex;align-items:center;gap:12px;padding:8px 4px;}
  button{font:600 13px sans-serif;border:1px solid #bbb;border-radius:8px;
         background:#fff;padding:6px 14px;cursor:pointer;}
  button:hover{background:#f0f0ef;}
  #t{color:var(--muted);font-variant-numeric:tabular-nums;}
  #legend{display:flex;flex-wrap:wrap;gap:10px 16px;padding:2px 4px 8px;}
  .lg{display:flex;align-items:center;gap:6px;color:var(--muted);
      cursor:pointer;user-select:none;}
  .lg.off{color:#b3b2ae;}
  .lg.off .sw{background:#cfcecb !important;}
  .sw{width:14px;height:3px;border-radius:2px;}
</style></head><body>
  <div id="bar">
    <button id="play">&#9654;&nbsp;Play</button>
    <span id="t">0.00 / 0.00 s</span>
    <span style="color:#8a8a86">&mdash; click the plot to seek, click a legend entry to hide/show a feature</span>
  </div>
  <div id="wrap">
    <canvas id="cv"></canvas>
    <div id="cursor"></div>
  </div>
  <div id="legend"></div>
  <audio id="au" src="__AUDIO__" preload="auto"></audio>
<script>
const D = __PAYLOAD__;
const W = D.width, WH = D.waveH, FH = D.featH, PH = D.partH, H = WH + FH + PH;
D.series.forEach((s,i)=>{ s.on = i < D.visible; });
D.showParts = true;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const dpr = window.devicePixelRatio || 1;
cv.width = W*dpr; cv.height = H*dpr; cv.style.height = H+'px'; ctx.scale(dpr,dpr);
function draw(){
  ctx.clearRect(0,0,W,H);
  // --- waveform (light grey) ---
  const midY = WH/2, amp = WH/2*0.92;
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--wave');
  ctx.lineWidth = 1; ctx.beginPath();
  for(let i=0;i<W;i++){
    ctx.moveTo(i+0.5, midY - D.wmax[i]*amp);
    ctx.lineTo(i+0.5, midY - D.wmin[i]*amp);
  }
  ctx.stroke();
  ctx.strokeStyle = '#ececea'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(0,WH+0.5); ctx.lineTo(W,WH+0.5); ctx.stroke();
  // --- feature curves ---
  const pad = 12, y0 = WH+pad, y1 = WH+FH-pad;
  for(const s of D.series){
    if(!s.on) continue;
    const n = s.v.length;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin='round';
    ctx.beginPath();
    for(let j=0;j<n;j++){
      const x = n>1 ? j/(n-1)*(W-2)+1 : W/2;
      const y = y1 - s.v[j]*(y1-y0);
      j? ctx.lineTo(x,y) : ctx.moveTo(x,y);
    }
    ctx.stroke();
  }
  // --- partial-gram (detected partial frequencies over time; grey = amplitude) ---
  if(PH > 0 && D.parts && D.showParts){
    const top = WH+FH, bot = H, ph = PH;
    ctx.strokeStyle = '#ececea'; ctx.beginPath();
    ctx.moveTo(0,top+0.5); ctx.lineTo(W,top+0.5); ctx.stroke();
    const dur = D.duration || 1, f0 = D.pf0, f1 = D.pf1;
    const P = D.parts, m = P.t.length, mk = 3;
    for(let i=0;i<m;i++){
      const x = P.t[i]/dur*W;
      const y = bot - (P.f[i]-f0)/(f1-f0)*ph;
      const g = Math.max(0, Math.min(1, P.a[i]));   // amplitude -> darkness
      ctx.fillStyle = 'rgba(0,0,0,'+(0.15+0.85*g).toFixed(3)+')';
      ctx.fillRect(x-mk/2, y-mk/2, mk, mk);
    }
    // y-axis frequency labels
    ctx.fillStyle = '#8a8a86'; ctx.font = '10px sans-serif';
    ctx.fillText(Math.round(f1)+' Hz', 3, top+11);
    ctx.fillText(Math.round(f0)+' Hz', 3, bot-3);
  }
}
draw();
const lg = document.getElementById('legend');
for(const s of D.series){
  const d=document.createElement('span'); d.className='lg';
  if(!s.on) d.classList.add('off');
  d.title='Click to hide/show this feature';
  d.innerHTML='<span class="sw" style="background:'+s.color+'"></span>'+s.name;
  d.onclick=()=>{ s.on=!s.on; d.classList.toggle('off', !s.on); draw(); };
  lg.appendChild(d);
}
if(PH > 0 && D.parts){
  const d=document.createElement('span'); d.className='lg';
  d.title='Click to hide/show the partial frequencies';
  d.innerHTML='<span class="sw" style="background:#666"></span>partials (grey = amplitude)';
  d.onclick=()=>{ D.showParts=!D.showParts; d.classList.toggle('off', !D.showParts); draw(); };
  lg.appendChild(d);
}
const au=document.getElementById('au'), cur=document.getElementById('cursor'),
      btn=document.getElementById('play'), tlab=document.getElementById('t');
cur.style.height = H+'px';
const dur = ()=> (isFinite(au.duration)&&au.duration>0)? au.duration : D.duration;
const fmt = x => x.toFixed(2);
function place(){
  const frac = Math.min(au.currentTime/dur(),1);
  cur.style.left = (frac*100)+'%';
  tlab.textContent = fmt(au.currentTime)+' / '+fmt(dur())+' s';
}
place();
let raf=null;
function loop(){ place(); raf=requestAnimationFrame(loop); }
btn.onclick=()=>{ if(au.paused){au.play();} else {au.pause();} };
au.onplay =()=>{ btn.innerHTML='&#10073;&#10073;&nbsp;Pause'; cancelAnimationFrame(raf); loop(); };
au.onpause=()=>{ btn.innerHTML='&#9654;&nbsp;Play'; cancelAnimationFrame(raf); place(); };
au.onended=()=>{ cancelAnimationFrame(raf); au.currentTime=0; place(); };
cv.style.cursor='pointer';
cv.onclick=(e)=>{ const r=cv.getBoundingClientRect();
  au.currentTime = Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*dur(); place(); };
</script></body></html>"""


def timbre_player(wav_path, features, visible=2, partials=None,
                  width=1000, wave_h=150, feat_h=210, partial_h=190):
    """Interactive feature player for Jupyter notebooks.

    Draws the waveform of ``wav_path`` in light grey with each feature overlaid
    as a coloured curve (normalised to [0, 1]), plus a play button and a
    vertical cursor that follows the audio. Legend entries are clickable to
    hide/show individual features; hidden entries are greyed out.

    Args:
        wav_path:  Path to the audio file (anything soundfile can read).
        features:  ``pandas.DataFrame`` with one column per feature -- e.g.
                   ``TimbreTrack().extract(wav).features`` -- or a result object
                   exposing a ``.features`` attribute.
        visible:   How many features are shown initially (the first ``visible``
                   columns; the rest start hidden and can be enabled by clicking
                   the legend). Pass ``None`` to show all.
        partials:  Optional long-format ``pandas.DataFrame`` with columns
                   ``[time_s, frequency, amplitude]`` (e.g.
                   ``WaveletRoughness().extract(wav).partials``). When given, an
                   extra panel shows the partial frequencies over time with the
                   grey level proportional to amplitude.
        width:     Internal plot width in pixels.
        wave_h:    Height of the waveform panel in pixels.
        feat_h:    Height of the feature panel in pixels.
        partial_h: Height of the partial-gram panel in pixels (only used when
                   ``partials`` is given).

    Returns:
        ``IPython.display.HTML`` -- renders inline in Jupyter.
    """
    from IPython.display import HTML   # local import: IPython only needed here

    if hasattr(features, "features"):  # accept a result object directly
        features = features.features

    # audio: waveform min/max envelope, one column per output pixel
    data, sr = sf.read(wav_path, always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / sr
    edges = np.linspace(0, len(mono), width + 1).astype(int)
    wmin = np.zeros(width); wmax = np.zeros(width)
    for i in range(width):
        seg = mono[edges[i]:edges[i + 1]]
        if seg.size:
            wmin[i] = seg.min(); wmax[i] = seg.max()
    peak = max(np.abs(wmin).max(), np.abs(wmax).max(), 1e-9)
    wmin /= peak; wmax /= peak

    # features -> [0, 1] (robust min/max), one coloured curve each
    palette = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4"]
    series = []
    for k, name in enumerate(features.columns):
        v = features[name].to_numpy().astype(float)
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = np.nanmin(v), np.nanmax(v)
            if hi <= lo:
                hi = lo + 1.0
        vn = np.nan_to_num(np.clip((v - lo) / (hi - lo), 0, 1))
        series.append({"name": str(name), "color": palette[k % len(palette)],
                       "v": [round(float(x), 4) for x in vn]})

    # optional partials panel
    part_h = 0
    parts = None
    pf0 = pf1 = 0.0
    if partials is not None and len(partials) > 0:
        part_h = partial_h
        pt = partials["time_s"].to_numpy(dtype=float)
        pfr = partials["frequency"].to_numpy(dtype=float)
        pa = partials["amplitude"].to_numpy(dtype=float)
        amax = max(float(np.nanmax(pa)), 1e-9)
        pa = np.clip(pa / amax, 0.0, 1.0)
        pf0 = float(np.nanmin(pfr)); pf1 = float(np.nanmax(pfr))
        if pf1 <= pf0:
            pf1 = pf0 + 1.0
        parts = {"t": [round(float(x), 4) for x in pt],
                 "f": [round(float(x), 2) for x in pfr],
                 "a": [round(float(x), 3) for x in pa]}

    # embed a small mono 22.05 kHz WAV as a data URI so it is self-contained
    target_sr = 22050
    if sr != target_sr:
        n_new = int(len(mono) * target_sr / sr)
        m = np.interp(np.linspace(0, len(mono), n_new, endpoint=False),
                      np.arange(len(mono)), mono)
    else:
        m = mono
    m = (m / max(np.abs(m).max(), 1e-9) * 0.98).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, m, target_sr, format="WAV", subtype="PCM_16")
    audio_uri = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()

    n_visible = len(series) if visible is None else max(0, int(visible))
    payload = json.dumps({
        "width": width, "waveH": wave_h, "featH": feat_h, "partH": part_h,
        "duration": duration, "visible": n_visible,
        "pf0": round(pf0, 2), "pf1": round(pf1, 2), "parts": parts,
        "wmin": [round(float(x), 3) for x in wmin],
        "wmax": [round(float(x), 3) for x in wmax],
        "series": series,
    })
    doc = _PLAYER_DOC.replace("__PAYLOAD__", payload).replace("__AUDIO__", audio_uri)
    total_h = wave_h + feat_h + part_h + 96
    iframe = ('<iframe style="width:100%;max-width:{w}px;height:{h}px;border:0;'
              'overflow:hidden" sandbox="allow-scripts allow-same-origin" '
              'srcdoc="{doc}"></iframe>').format(
                  w=width + 24, h=total_h, doc=escape(doc, quote=True))
    # wrapped in a <div> so IPython.display.HTML does not emit its
    # "Consider using IPython.display.IFrame instead" warning
    return HTML("<div>" + iframe + "</div>")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _embed_audio_datauri(mono, sr, target_sr=22050):
    """Return a self-contained ``data:audio/wav`` URI for a mono signal."""
    if sr != target_sr:
        n_new = int(len(mono) * target_sr / sr)
        m = np.interp(np.linspace(0, len(mono), n_new, endpoint=False),
                      np.arange(len(mono)), mono)
    else:
        m = mono
    m = (m / max(np.abs(m).max(), 1e-9) * 0.98).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, m, target_sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _waveform_envelope(mono, width):
    """Return (wmin, wmax) min/max envelopes, one value per output pixel."""
    edges = np.linspace(0, len(mono), width + 1).astype(int)
    wmin = np.zeros(width)
    wmax = np.zeros(width)
    for i in range(width):
        seg = mono[edges[i]:edges[i + 1]]
        if seg.size:
            wmin[i] = seg.min()
            wmax[i] = seg.max()
    peak = max(np.abs(wmin).max(), np.abs(wmax).max(), 1e-9)
    return wmin / peak, wmax / peak


# ---------------------------------------------------------------------------
# Pitch player
# ---------------------------------------------------------------------------

_PITCH_DOC = r"""<!doctype html><html><head><meta charset="utf-8"><style>
  :root{--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--wave:#c9c9c7;}
  body{margin:0;font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;
       color:var(--ink);background:var(--surface);}
  #wrap{position:relative;}
  canvas{display:block;width:100%;}
  #cursor{position:absolute;top:0;width:2px;background:#0b0b0b;opacity:.55;
          pointer-events:none;left:0;display:none;}
  #bar{display:flex;align-items:center;gap:8px;padding:8px 4px;flex-wrap:wrap;}
  button{font:600 13px sans-serif;border:1px solid #bbb;border-radius:8px;
         background:#fff;padding:6px 12px;cursor:pointer;}
  button:hover{background:#f0f0ef;}
  #t{color:var(--muted);font-variant-numeric:tabular-nums;}
  #legend{display:flex;flex-wrap:wrap;gap:10px 16px;padding:2px 4px 8px;}
  .lg{display:flex;align-items:center;gap:6px;color:var(--muted);
      cursor:pointer;user-select:none;}
  .lg.off{color:#b3b2ae;}
  .lg.off .sw{background:#cfcecb !important;}
  .sw{width:14px;height:3px;border-radius:2px;}
  .hint{color:#8a8a86;}
</style></head><body>
  <div id="bar">
    <button id="play">&#9654;&nbsp;Play</button>
    <span id="t">0.00 / 0.00 s</span>
    <button id="zin">Zoom +</button>
    <button id="zout">Zoom &minus;</button>
    <button id="zreset">Reset</button>
    <span class="hint">&mdash; wheel = zoom, drag = pan, click = seek</span>
  </div>
  <div id="wrap">
    <canvas id="cv"></canvas>
    <div id="cursor"></div>
  </div>
  <div id="legend"></div>
  <audio id="au" src="__AUDIO__" preload="auto"></audio>
<script>
const D = __PAYLOAD__;
const W = D.width, WH = D.waveH, PH = D.pitchH, H = WH + PH;
const layers = {waveform:true, impulses:true};
D.tracks.forEach(function(tr){ tr.on = true; });
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const dpr = window.devicePixelRatio || 1;
cv.width = W*dpr; cv.height = H*dpr; cv.style.height = H+'px'; ctx.scale(dpr,dpr);
const lf0 = Math.log(D.fmin), lf1 = Math.log(D.fmax);
let T0 = 0, T1 = D.duration;                 // visible time window [s]
let samples = null, sr = 44100, peak = 1;    // decoded audio (for zoom)

function xOf(t){ return (t - T0) / (T1 - T0) * W; }
function tOf(x){ return T0 + x / W * (T1 - T0); }
function yFreq(f){ return WH + PH - (Math.log(f)-lf0)/(lf1-lf0)*PH; }

function clampView(){
  let span = T1 - T0;
  if(span > D.duration){ span = D.duration; }
  if(span < 0.002){ span = 0.002; }
  if(T0 < 0){ T0 = 0; T1 = span; }
  if(T1 > D.duration){ T1 = D.duration; T0 = T1 - span; if(T0 < 0) T0 = 0; }
}

function drawWaveSamples(){
  const i0 = Math.max(0, Math.floor(T0*sr)), i1 = Math.min(samples.length, Math.ceil(T1*sr));
  const nSamp = i1 - i0, spp = nSamp / W, mid = WH/2, amp = WH/2*0.92;
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--wave');
  ctx.lineWidth = 1; ctx.beginPath();
  if(spp >= 2){
    for(let c=0;c<W;c++){
      const a = i0 + Math.floor(c*spp), b = Math.min(i1, i0 + Math.floor((c+1)*spp));
      let mn = 1e9, mx = -1e9;
      for(let i=a;i<b;i++){ const v = samples[i]; if(v<mn)mn=v; if(v>mx)mx=v; }
      if(mn>mx){ mn = mx = 0; }
      ctx.moveTo(c+0.5, mid - mx/peak*amp);
      ctx.lineTo(c+0.5, mid - mn/peak*amp);
    }
  } else {
    for(let i=i0;i<i1;i++){
      const x = (i/sr - T0)/(T1-T0)*W, y = mid - samples[i]/peak*amp;
      (i===i0)? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
  }
  ctx.stroke();
}

function drawWaveCoarse(){
  const mid = WH/2, amp = WH/2*0.92;
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--wave');
  ctx.lineWidth = 1; ctx.beginPath();
  const c0 = T0/D.duration*W, c1 = T1/D.duration*W;
  for(let c=0;c<W;c++){
    const src = Math.min(W-1, Math.max(0, Math.round(c0 + c/W*(c1-c0))));
    ctx.moveTo(c+0.5, mid - D.wmax[src]*amp);
    ctx.lineTo(c+0.5, mid - D.wmin[src]*amp);
  }
  ctx.stroke();
}

function draw(){
  ctx.clearRect(0,0,W,H);
  if(layers.waveform){ samples ? drawWaveSamples() : drawWaveCoarse(); }
  // impulses as vertical lines over the waveform (grey area), opacity ~ amplitude
  if(layers.impulses && D.impulses){
    const it = D.impulses.t, ia = D.impulses.a, m = it.length;
    ctx.lineWidth = 1;
    for(let i=0;i<m;i++){
      if(it[i] < T0 || it[i] > T1) continue;
      const x = xOf(it[i]);
      ctx.strokeStyle = 'rgba(217,72,15,' + (0.22 + 0.7*ia[i]).toFixed(3) + ')';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, WH); ctx.stroke();
    }
  }
  // panel separator
  ctx.strokeStyle = '#ececea'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0,WH+0.5); ctx.lineTo(W,WH+0.5); ctx.stroke();
  // pitch panel: octave gridlines + labels
  ctx.fillStyle = '#a9a8a4'; ctx.font = '10px sans-serif';
  ctx.strokeStyle = '#eeeeec';
  let gf = Math.pow(2, Math.ceil(Math.log2(D.fmin)));
  for(; gf < D.fmax; gf *= 2){
    const y = yFreq(gf);
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
    ctx.fillText(Math.round(gf)+' Hz', 3, y-2);
  }
  // f0 tracks (break where f<=0 or outside view)
  for(const tr of D.tracks){
    if(!tr.on) continue;
    ctx.strokeStyle = tr.color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    const t = tr.t, f = tr.f, n = t.length;
    let started = false;
    for(let i=0;i<n;i++){
      if(t[i] < T0 || t[i] > T1 || f[i] <= 0){ if(started){ ctx.stroke(); started=false; } continue; }
      const x = xOf(t[i]);
      const fc = Math.max(D.fmin, Math.min(D.fmax, f[i]));
      const y = yFreq(fc);
      if(!started){ ctx.beginPath(); ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y);
    }
    if(started) ctx.stroke();
  }
  placeCursor();
}
let _raf = null;
function scheduleDraw(){ if(_raf) return; _raf = requestAnimationFrame(function(){ _raf=null; draw(); }); }

// --- legend ---
const lg = document.getElementById('legend');
function addChip(name, color, getOn, toggle){
  const d = document.createElement('span'); d.className = 'lg';
  if(!getOn()) d.classList.add('off');
  d.title = 'Click to hide/show';
  d.innerHTML = '<span class="sw" style="background:'+color+'"></span>'+name;
  d.onclick = function(){ toggle(); d.classList.toggle('off', !getOn()); scheduleDraw(); };
  lg.appendChild(d);
}
addChip('waveform', '#c9c9c7', function(){return layers.waveform;}, function(){layers.waveform=!layers.waveform;});
for(const tr of D.tracks){ (function(tr){
  addChip(tr.name, tr.color, function(){return tr.on;}, function(){tr.on=!tr.on;});
})(tr); }
if(D.impulses){ addChip('impulses', '#d9480f', function(){return layers.impulses;}, function(){layers.impulses=!layers.impulses;}); }

// --- audio + cursor ---
const au = document.getElementById('au'), cur = document.getElementById('cursor'),
      btn = document.getElementById('play'), tlab = document.getElementById('t');
cur.style.height = H+'px';
const durf = function(){ return (isFinite(au.duration)&&au.duration>0)? au.duration : D.duration; };
const fmt = function(x){ return x.toFixed(2); };
function placeCursor(){
  const ct = au.currentTime;
  if(ct >= T0 && ct <= T1){ cur.style.display='block'; cur.style.left = (xOf(ct)/W*100)+'%'; }
  else { cur.style.display='none'; }
  tlab.textContent = fmt(ct)+' / '+fmt(durf())+' s';
}
let _craf = null;
function loop(){ placeCursor(); _craf = requestAnimationFrame(loop); }
btn.onclick = function(){ if(au.paused) au.play(); else au.pause(); };
au.onplay = function(){ btn.innerHTML='&#10073;&#10073;&nbsp;Pause'; cancelAnimationFrame(_craf); loop(); };
au.onpause = function(){ btn.innerHTML='&#9654;&nbsp;Play'; cancelAnimationFrame(_craf); placeCursor(); };
au.onended = function(){ cancelAnimationFrame(_craf); au.currentTime=0; placeCursor(); };

// --- zoom / pan / seek ---
function zoomAt(tc, factor){
  let nT0 = tc - (tc - T0)*factor, nT1 = tc + (T1 - tc)*factor;
  T0 = nT0; T1 = nT1; clampView(); scheduleDraw();
}
cv.addEventListener('wheel', function(e){
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const tc = tOf((e.clientX - r.left)/r.width*W);
  zoomAt(tc, e.deltaY < 0 ? 0.8 : 1.25);
}, {passive:false});
document.getElementById('zin').onclick = function(){ zoomAt((T0+T1)/2, 0.6); };
document.getElementById('zout').onclick = function(){ zoomAt((T0+T1)/2, 1.7); };
document.getElementById('zreset').onclick = function(){ T0=0; T1=D.duration; scheduleDraw(); };
let dragging=false, dragMoved=false, lastX=0;
cv.addEventListener('mousedown', function(e){ dragging=true; dragMoved=false; lastX=e.clientX; });
window.addEventListener('mousemove', function(e){
  if(!dragging) return;
  const r = cv.getBoundingClientRect();
  const dx = (e.clientX - lastX)/r.width*W; lastX = e.clientX;
  if(Math.abs(dx) > 0.5) dragMoved = true;
  const dt = dx/W*(T1-T0); T0 -= dt; T1 -= dt; clampView(); scheduleDraw();
});
window.addEventListener('mouseup', function(e){
  if(dragging && !dragMoved){
    const r = cv.getBoundingClientRect();
    au.currentTime = Math.max(0, Math.min(D.duration, tOf((e.clientX - r.left)/r.width*W)));
    placeCursor();
  }
  dragging = false;
});
cv.style.cursor = 'crosshair';

// --- decode embedded audio for sample-accurate zoom ---
const AC = window.AudioContext || window.webkitAudioContext;
if(AC){
  try{
    const ac = new AC();
    fetch(au.src).then(function(r){ return r.arrayBuffer(); })
      .then(function(b){ return ac.decodeAudioData(b); })
      .then(function(buf){
        samples = buf.getChannelData(0); sr = buf.sampleRate;
        let mx = 1e-9; for(let i=0;i<samples.length;i+=7){ const v=Math.abs(samples[i]); if(v>mx)mx=v; }
        peak = mx; scheduleDraw();
      }).catch(function(){});
  }catch(e){}
}
draw();
</script></body></html>"""


def pitch_player(wav_path, result, impulses=None, pitch_col="Pitch",
                 fmin=None, fmax=None, width=1000, wave_h=190, pitch_h=250):
    """Interactive f0 / pitch player for Jupyter notebooks, with zoom.

    Draws the waveform of ``wav_path`` in light grey and the fundamental
    frequency (f0) track on a logarithmic frequency axis, with a play button
    and a vertical cursor that follows the audio. Unvoiced frames (f0 <= 0)
    leave a gap. The plot supports **horizontal zoom** (mouse wheel or the
    Zoom buttons), **pan** (drag) and **seek** (click); the waveform is
    re-rendered from the decoded audio down to sample level.

    Args:
        wav_path:   Path to the audio file.
        result:     A ``PitchTrack`` result (uses ``.features``) or a DataFrame
                    indexed by ``time_s`` containing a pitch column.
        impulses:   Optional ``ImpulsePattern`` result or a DataFrame with
                    ``[time_s, amplitude]``. When given, the impulses are drawn
                    as vertical lines over the waveform (opacity ~ amplitude),
                    toggleable via the legend.
        pitch_col:  Name of the f0 column (default ``"Pitch"``).
        fmin, fmax: Frequency-axis limits in Hz; by default the voiced f0 range.
        width:      Internal plot width in pixels.
        wave_h:     Height of the waveform panel in pixels.
        pitch_h:    Height of the pitch panel in pixels.

    Returns:
        ``IPython.display.HTML`` -- renders inline in Jupyter.
    """
    from IPython.display import HTML

    features = result.features if hasattr(result, "features") else result
    times = np.asarray(features.index, dtype=float)
    f0 = features[pitch_col].to_numpy(dtype=float)

    data, sr = sf.read(wav_path, always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / sr
    wmin, wmax = _waveform_envelope(mono, width)
    audio_uri = _embed_audio_datauri(mono, sr)

    voiced = f0[f0 > 0]
    if fmin is None:
        fmin = float(np.percentile(voiced, 1) / 1.12) if voiced.size else 50.0
    if fmax is None:
        fmax = float(np.percentile(voiced, 99) * 1.12) if voiced.size else 1000.0
    fmin = max(1.0, fmin)
    if fmax <= fmin:
        fmax = fmin * 2.0

    tracks = [{"name": "f0 (" + str(pitch_col) + ")", "color": "#2a78d6",
               "t": [round(float(x), 4) for x in times],
               "f": [round(float(x), 3) for x in f0]}]

    imp_payload = None
    if impulses is not None:
        imp_df = impulses.impulses if hasattr(impulses, "impulses") else impulses
        if len(imp_df) > 0:
            it = imp_df["time_s"].to_numpy(dtype=float)
            ia = imp_df["amplitude"].to_numpy(dtype=float)
            amax = max(float(np.nanmax(ia)), 1e-12)
            ia = np.clip(ia / amax, 0.0, 1.0)
            imp_payload = {"t": [round(float(x), 5) for x in it],
                           "a": [round(float(x), 3) for x in ia]}

    payload = json.dumps({
        "width": width, "waveH": wave_h, "pitchH": pitch_h,
        "duration": duration, "fmin": round(fmin, 3), "fmax": round(fmax, 3),
        "wmin": [round(float(x), 3) for x in wmin],
        "wmax": [round(float(x), 3) for x in wmax],
        "tracks": tracks, "impulses": imp_payload,
    })
    doc = _PITCH_DOC.replace("__PAYLOAD__", payload).replace("__AUDIO__", audio_uri)
    total_h = wave_h + pitch_h + 120
    iframe = ('<iframe style="width:100%;max-width:{w}px;height:{h}px;border:0;'
              'overflow:hidden" sandbox="allow-scripts allow-same-origin" '
              'srcdoc="{doc}"></iframe>').format(
                  w=width + 24, h=total_h, doc=escape(doc, quote=True))
    return HTML("<div>" + iframe + "</div>")
