/* Hero explainer: a forward-moving camera, its image, and what happens at the image border.
   One synthetic world drawn two ways: a top view (left, or top on phones) and the image of
   camera t (right, or bottom) with the space beyond the image border. Every position is a
   pinhole projection of the same 3D points, so the orange and green marks are exact for this
   scene. Phases: 0 drive, 1 frames t-1/t/t+1 and matches, 2 out of frame, 3 seen in t-1,
   4 reprojected rigid flow. prefers-reduced-motion: no autoplay, a static final frame. */
(function () {
  "use strict";
  var fig = document.querySelector("[data-explainer]");
  if (!fig) return;
  var canvas = fig.querySelector("canvas");
  var ctx = canvas.getContext("2d");
  if (!ctx) return;
  var stepBtns = Array.prototype.slice.call(fig.querySelectorAll("[data-phase]"));
  var playBtn = fig.querySelector("[data-play]");
  var mqReduce = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ---------------------------------------------------------------- world (metres) */
  /* camera coordinates: x right, y down, z forward; camera t sits at z = 0 */
  var V = 0.9;            // camera displacement per frame
  var CAM_H = 1.65;       // camera height above the road
  var DRIVE = 6.3;        // distance driven during phase 0 (seven frames)
  var ASPECT = 2.6;       // image width / height
  var HOR = 0.42;         // horizon position, fraction of the image height from the top
  var NA = HOR * 2 / ASPECT, NB = (1 - HOR) * 2 / ASPECT;   // y/z bounds of the image
  var X = { x: 4.2, y: 0.65, z: 4.4 };                       // the highlighted fence point

  var s = 11;
  function rnd() { s = (s * 16807) % 2147483647; return s / 2147483647; }

  var lines = [];     // {p: [[x,y,z],...], w: weight}
  var pts = [];       // tracked scene points {x,y,z}
  var cars = [], trees = [];

  function poly(arr, w) { lines.push({ p: arr, w: w || 1 }); }
  var z, k, i;
  // road edges and lane dashes
  for (var side = -1; side <= 1; side += 2) {
    var e = [];
    for (z = -10; z <= 90; z += 1) e.push([side * 3.5, CAM_H, z]);
    poly(e, 1.2);
  }
  for (k = -3; k < 16; k++) poly([[0, CAM_H, k * 6], [0, CAM_H, k * 6 + 3]], 1.6);
  // right fence: rails, posts, and points on it
  var rb = [], rt = [];
  for (z = -10; z <= 70; z += 1) { rb.push([4.2, CAM_H, z]); rt.push([4.2, 0.35, z]); }
  poly(rb, 0.9); poly(rt, 1.1);
  for (z = -10; z <= 70; z += 2) poly([[4.2, CAM_H, z], [4.2, 0.2, z]], 1.1);
  for (z = 0.4; z <= 40; z += 1.0) { pts.push({ x: 4.2, y: 0.65, z: z }); pts.push({ x: 4.2, y: 1.15, z: z + 0.5 }); }
  // parked cars on the left (near face at x = -4.5)
  [6.5, 15.5, 27, 38].forEach(function (z0) {
    var L = 4.3, x0 = -4.5, top = CAM_H - 1.45, belt = CAM_H - 0.85, bot = CAM_H - 0.25;
    cars.push({ x0: x0 - 1.8, x1: x0, z0: z0, z1: z0 + L });
    poly([[x0, bot, z0], [x0, bot, z0 + L], [x0, belt, z0 + L], [x0, belt, z0]].concat([[x0, bot, z0]]), 1.2);
    poly([[x0, belt, z0 + 0.6], [x0 - 0.2, top, z0 + 1.3], [x0 - 0.2, top, z0 + 3.2], [x0, belt, z0 + 3.9]], 1.2);
    for (i = 0; i < 6; i++) { pts.push({ x: x0, y: belt + 0.3, z: z0 + 0.4 + i * 0.7 }); }
    for (i = 0; i < 3; i++) { pts.push({ x: x0 - 0.15, y: top + 0.25, z: z0 + 1.5 + i * 0.8 }); }
  });
  // building facades on the left, with windows
  [[-2, 14], [17, 31], [34, 52], [55, 75]].forEach(function (b) {
    var xf = -8.5, h = CAM_H - 7.5 - rnd() * 2;
    poly([[xf, CAM_H, b[0]], [xf, h, b[0]], [xf, h, b[1]], [xf, CAM_H, b[1]]], 1.1);
    for (var zz = b[0] + 1.5; zz < b[1] - 1.5; zz += 3) {
      for (var yy = CAM_H - 2.4; yy > h + 1; yy -= 2.6) {
        poly([[xf, yy, zz], [xf, yy - 1.2, zz], [xf, yy - 1.2, zz + 1.2], [xf, yy, zz + 1.2], [xf, yy, zz]], 0.7);
        pts.push({ x: xf, y: yy, z: zz });
      }
    }
  });
  // trees behind the fence
  for (z = 3; z < 60; z += 5 + rnd() * 3) {
    var tx = 6.6 + rnd() * 3, r = 1.3 + rnd() * 0.9, cy = -1.8 - rnd() * 1.6;
    trees.push({ x: tx, z: z, r: r });
    poly([[tx, CAM_H, z], [tx, cy + r * 0.6, z]], 1.3);
    for (i = 0; i < 9; i++) {
      var a = rnd() * Math.PI * 2, q = Math.sqrt(rnd()) * r;
      pts.push({ x: tx + Math.cos(a) * q * 0.9, y: cy + Math.sin(a) * q * 0.8, z: z + (rnd() - 0.5) * r });
    }
  }
  // road texture
  for (i = 0; i < 46; i++) pts.push({ x: -3.2 + rnd() * 6.4, y: CAM_H, z: 4 + Math.pow(rnd(), 1.6) * 26 });
  pts.push({ x: X.x, y: X.y, z: X.z, isX: true });

  function proj(p, zc) {
    var Z = p.z - zc;
    if (Z < 0.25) return null;
    return { u: p.x / Z, v: p.y / Z, Z: Z };
  }
  function inside(q) { return q && q.u >= -1 && q.u <= 1 && q.v >= -NA && q.v <= NB; }

  // classify tracked points at the frozen time t (camera at z = 0)
  var tracked = [];
  pts.forEach(function (p, idx) {
    var a = proj(p, 0), b = proj(p, V), c = proj(p, -V);
    if (!inside(a) || a.Z < 4 || a.Z > 34) return;
    var oof = !inside(b);
    tracked.push({ p: p, t: a, n: b, b: c, oof: oof, prev: inside(c), show: oof || p.isX || idx % 2 === 0 });
  });

  /* ---------------------------------------------------------------- timing */
  var DUR = [3.4, 2.6, 2.8, 2.8, 3.6];
  var HOLD = 1.6, FADE = 0.7;
  var START = [0]; for (i = 0; i < DUR.length; i++) START.push(START[i] + DUR[i]);
  var TOTAL = START[DUR.length] + HOLD + FADE;
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function easeOut(t) { t = clamp(t, 0, 1); return 1 - Math.pow(1 - t, 3); }
  function easeInOut(t) { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

  /* state at time T (seconds into the loop): camera position and the alpha of each layer */
  function stateAt(T) {
    var st = { zc: 0, a: [0, 0, 0, 0, 0], g: 1, phase: 0, prog: 0 };
    st.zc = -DRIVE * (1 - easeInOut(T / DUR[0]));
    for (var p = 1; p < DUR.length; p++) st.a[p] = easeOut((T - START[p]) / 0.8);
    st.a[0] = 1;
    // the growth of the reprojected arrows in phase 4
    st.grow = easeInOut((T - START[4] - 0.2) / 1.6);
    st.back = easeInOut((T - START[3] - 0.2) / 1.1);
    st.g = Math.min(easeOut(T / 0.5), T > TOTAL - FADE ? 1 - easeOut((T - (TOTAL - FADE)) / FADE) : 1);
    for (p = DUR.length - 1; p >= 0; p--) if (T >= START[p]) { st.phase = p; st.prog = clamp((T - START[p]) / DUR[p], 0, 1); break; }
    if (T >= START[DUR.length]) { st.phase = DUR.length - 1; st.prog = 1; }
    return st;
  }

  /* ---------------------------------------------------------------- colours */
  var C = {};
  function readColours() {
    var cs = getComputedStyle(document.documentElement);
    function v(n) { return cs.getPropertyValue(n).trim(); }
    C = { ink: v("--ink"), ink2: v("--ink-2"), ink3: v("--ink-3"), rule: v("--rule"), soft: v("--rule-soft"),
      sunken: v("--sunken"), surface: v("--surface"), net: v("--net"), fill: v("--fill"),
      oof: v("--oof-mark"), oofText: v("--oof"), bg: v("--bg") };
  }

  /* ---------------------------------------------------------------- layout */
  var L = {};
  var dpr = 1;
  function layout() {
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return false;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    var portrait = w / h < 1.3;          // the CSS gives phones a tall canvas
    var sc = clamp(w / 1100, 0.62, 1.15);
    if (portrait) sc = clamp(w / 520, 0.62, 1);
    L = { w: w, h: h, portrait: portrait, sc: sc };
    var pad = 14 * sc;
    if (portrait) {
      var th = Math.round(h * 0.47);
      L.top = { x: 0, y: 0, w: w, h: th };
      L.img = { x: 0, y: th, w: w, h: h - th };
    } else {
      var tw = Math.round(w * 0.31);
      L.top = { x: 0, y: 0, w: tw, h: h };
      L.img = { x: tw, y: 0, w: w - tw, h: h };
    }
    // the image frame inside its panel, with room around it for what lies beyond the border
    var fw = Math.min(L.img.w * (portrait ? 0.74 : 0.74), (L.img.h - 2 * pad) * ASPECT * 0.66);
    var fh = fw / ASPECT;
    L.fr = { x: L.img.x + (L.img.w - fw) / 2, y: L.img.y + (L.img.h - fh) / 2 + (portrait ? 4 : -6) * sc, w: fw, h: fh };
    // top view: metres -> pixels, camera t near the bottom
    var mx = portrait ? 6.6 : 5.6;
    L.m = L.top.w / (2 * mx);
    L.tvx = L.top.x + L.top.w / 2;
    L.tvy = L.top.y + L.top.h - V * L.m - (portrait ? 20 : 40) * sc;
    return true;
  }
  function imgPx(q) {
    var f = L.fr;
    return [f.x + (q.u + 1) / 2 * f.w, f.y + (q.v + NA) / (NA + NB) * f.h];
  }
  function topPx(x, z, zc) { return [L.tvx + x * L.m, L.tvy - (z - zc) * L.m]; }

  /* ---------------------------------------------------------------- drawing helpers */
  function arrow(x1, y1, x2, y2, col, lw, head) {
    var dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
    if (len < 0.5) return;
    var ux = dx / len, uy = dy / len, hs = Math.min(head, len * 0.6);
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - ux * hs * 0.6, y2 - uy * hs * 0.6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - ux * hs - uy * hs * 0.5, y2 - uy * hs + ux * hs * 0.5);
    ctx.lineTo(x2 - ux * hs + uy * hs * 0.5, y2 - uy * hs - ux * hs * 0.5);
    ctx.closePath(); ctx.fill();
  }
  function fpx(size) { return Math.max(11, Math.round(size * L.sc)); }
  function label(text, x, y, col, size, align, weight) {
    ctx.font = (weight || 500) + " " + fpx(size) + "px Barlow, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = col; ctx.textAlign = align || "left"; ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
  }
  function italicLabel(base, sup, x, y, col, size, align) {
    // "t", "t-1", "t+1" with an italic t, as in the paper
    ctx.font = "italic 500 " + fpx(size) + "px 'Source Serif 4', Georgia, serif";
    var w1 = ctx.measureText(base).width;
    ctx.font = "500 " + fpx(size) + "px Barlow, Arial, sans-serif";
    var w2 = ctx.measureText(sup).width;
    var x0 = align === "right" ? x - w1 - w2 : align === "center" ? x - (w1 + w2) / 2 : x;
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = col;
    ctx.font = "italic 500 " + fpx(size) + "px 'Source Serif 4', Georgia, serif";
    ctx.fillText(base, x0, y);
    ctx.font = "500 " + fpx(size) + "px Barlow, Arial, sans-serif";
    ctx.fillText(sup, x0 + w1, y);
  }
  function withAlpha(a, fn) { if (a <= 0.001) return; ctx.save(); ctx.globalAlpha *= a; fn(); ctx.restore(); }
  function clipRect(r) { ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip(); }

  /* ---------------------------------------------------------------- top view */
  function drawTop(st) {
    var zc = st.zc, m = L.m, sc = L.sc;
    ctx.save(); clipRect(L.top);
    ctx.fillStyle = C.surface; ctx.fillRect(L.top.x, L.top.y, L.top.w, L.top.h);
    // road
    var a = topPx(-3.5, zc + 40, zc), b = topPx(3.5, zc - 20, zc);
    ctx.fillStyle = C.sunken; ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1.2 * sc;
    ctx.setLineDash([8 * sc, 10 * sc]);
    ctx.lineDashOffset = -((zc * m) % (18 * sc));
    ctx.beginPath(); var c0 = topPx(0, zc + 40, zc), c1 = topPx(0, zc - 20, zc); ctx.moveTo(c0[0], c0[1]); ctx.lineTo(c1[0], c1[1]); ctx.stroke();
    ctx.setLineDash([]);
    // fence with posts
    var f0 = topPx(4.2, zc + 40, zc), f1 = topPx(4.2, zc - 20, zc);
    ctx.strokeStyle = C.ink3; ctx.lineWidth = 2 * sc;
    ctx.beginPath(); ctx.moveTo(f0[0], f0[1]); ctx.lineTo(f1[0], f1[1]); ctx.stroke();
    ctx.fillStyle = C.ink3;
    for (var zz = Math.floor((zc - 20) / 2) * 2; zz < zc + 40; zz += 2) { var pp = topPx(4.2, zz, zc); ctx.fillRect(pp[0] - 2 * sc, pp[1] - 2 * sc, 4 * sc, 4 * sc); }
    // cars and trees
    ctx.strokeStyle = C.ink3; ctx.lineWidth = 1.2 * sc;
    cars.forEach(function (c) {
      var p0 = topPx(c.x0, c.z1, zc), p1 = topPx(c.x1, c.z0, zc);
      roundRect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], 4 * sc); ctx.stroke();
    });
    trees.forEach(function (t) {
      var p = topPx(t.x, t.z, zc);
      ctx.beginPath(); ctx.arc(p[0], p[1], t.r * m, 0, Math.PI * 2); ctx.stroke();
    });
    // the highlighted point X
    var Xp = topPx(X.x, X.z, zc);
    var cams = [{ z: -V, lab: "−1", a: st.a[1] }, { z: 0, lab: "", a: 1 }, { z: V, lab: "+1", a: st.a[1] }];
    // rays (drawn under the cameras)
    withAlpha(st.a[3], function () {
      [cams[0], cams[1]].forEach(function (c) {
        var cp = topPx(0, c.z, zc);
        ctx.strokeStyle = C.fill; ctx.lineWidth = 2 * sc;
        ctx.beginPath(); ctx.moveTo(cp[0], cp[1]); ctx.lineTo(Xp[0], Xp[1]); ctx.stroke();
      });
    });
    var c1p = topPx(0, V, zc);
    var fl = 0.42;                          // focal length of the drawn frustum, metres
    // where X lands on camera t+1's image line (outside its extent)
    var hitx = X.x * fl / (X.z - V), hit = topPx(hitx, V + fl, zc);
    withAlpha(st.a[2] * (1 - 0.6 * st.a[4]), function () {
      ctx.strokeStyle = C.oof; ctx.lineWidth = 2 * sc; ctx.setLineDash([5 * sc, 4 * sc]);
      ctx.beginPath(); ctx.moveTo(c1p[0], c1p[1]); ctx.lineTo(Xp[0], Xp[1]); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); var e = topPx(fl, V + fl, zc); ctx.moveTo(e[0], e[1]); ctx.lineTo(hit[0], hit[1]);
      ctx.setLineDash([2 * sc, 3 * sc]); ctx.stroke(); ctx.setLineDash([]);
      cross(hit[0], hit[1], 5 * sc, C.oof);
    });
    withAlpha(st.a[4], function () {
      ctx.strokeStyle = C.fill; ctx.lineWidth = 2.2 * sc;
      ctx.beginPath(); ctx.moveTo(Xp[0], Xp[1]); ctx.lineTo(Xp[0] + (c1p[0] - Xp[0]) * st.grow, Xp[1] + (c1p[1] - Xp[1]) * st.grow); ctx.stroke();
      if (st.grow > 0.98) { ctx.fillStyle = C.fill; ctx.beginPath(); ctx.arc(hit[0], hit[1], 4.2 * sc, 0, Math.PI * 2); ctx.fill(); }
    });
    // cameras: frustum triangle + image line; FOV wedge of the current camera in phase 0
    cams.forEach(function (c, idx) {
      var alpha = idx === 1 ? 1 : c.a;
      withAlpha(alpha, function () {
        var zz = idx === 1 ? zc : c.z;
        var p = topPx(0, zz, zc);
        var e1 = topPx(-fl, zz + fl, zc), e2 = topPx(fl, zz + fl, zc);
        // field of view (90 degrees): camera t while driving, camera t+1 afterwards
        withAlpha(idx === 2 ? 1 : idx === 1 ? 1 - st.a[1] : 0, function () {
          ctx.strokeStyle = C.ink3; ctx.lineWidth = 1 * sc; ctx.setLineDash([4 * sc, 4 * sc]);
          var R = 18;
          var w1 = topPx(-R, zz + R, zc), w2 = topPx(R, zz + R, zc);
          ctx.beginPath(); ctx.moveTo(w1[0], w1[1]); ctx.lineTo(p[0], p[1]); ctx.lineTo(w2[0], w2[1]); ctx.stroke(); ctx.setLineDash([]);
        });
        ctx.fillStyle = C.surface; ctx.strokeStyle = idx === 1 ? C.ink : C.ink2; ctx.lineWidth = 1.6 * sc;
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(e1[0], e1[1]); ctx.lineTo(e2[0], e2[1]); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.strokeStyle = C.net; ctx.lineWidth = 3.2 * sc;
        ctx.beginPath(); ctx.moveTo(e1[0], e1[1]); ctx.lineTo(e2[0], e2[1]); ctx.stroke();
        if (st.a[1] > 0) italicLabel("t", c.lab, p[0] - 12 * sc, p[1] - 2 * sc, idx === 1 ? C.ink : C.ink2, 15, "right");
      });
    });
    // image points of X on t-1 and t (phase 3)
    withAlpha(st.a[3], function () {
      [cams[0], cams[1]].forEach(function (c) {
        var ix = X.x * fl / (X.z - c.z), ip = topPx(ix, c.z + fl, zc);
        ctx.fillStyle = C.fill; ctx.beginPath(); ctx.arc(ip[0], ip[1], 3.6 * sc, 0, Math.PI * 2); ctx.fill();
      });
    });
    // X
    withAlpha(st.a[1], function () {
      ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(Xp[0], Xp[1], 4.6 * sc, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2 * sc; ctx.beginPath(); ctx.arc(Xp[0], Xp[1], 9 * sc, 0, Math.PI * 2); ctx.stroke();
      ctx.font = "italic 600 " + fpx(16) + "px 'Source Serif 4', Georgia, serif";
      ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText("X", Xp[0] + 12 * sc, Xp[1] - 6 * sc);
    });
    ctx.font = "600 " + fpx(14) + "px Barlow, Arial, sans-serif";
    var tw = ctx.measureText("Top view").width;
    ctx.fillStyle = C.surface; ctx.globalAlpha = 0.92;
    roundRect(L.top.x + 8 * sc, L.top.y + 8 * sc, tw + 14 * sc, fpx(14) + 12 * sc, 5 * sc); ctx.fill(); ctx.globalAlpha = 1;
    label("Top view", L.top.x + 15 * sc, L.top.y + 14 * sc + fpx(14) / 2, C.ink2, 14, "left", 600);
    ctx.restore();
    // divider between the two views
    ctx.strokeStyle = C.soft; ctx.lineWidth = 1;
    ctx.beginPath();
    if (L.portrait) { ctx.moveTo(0, L.img.y + 0.5); ctx.lineTo(L.w, L.img.y + 0.5); } else { ctx.moveTo(L.img.x + 0.5, 0); ctx.lineTo(L.img.x + 0.5, L.h); }
    ctx.stroke();
  }
  function roundRect(x, y, w, h, r) {
    if (w < 0) { x += w; w = -w; } if (h < 0) { y += h; h = -h; }
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function cross(x, y, r, col) {
    ctx.strokeStyle = col; ctx.lineWidth = 2 * L.sc;
    ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke();
  }

  /* ---------------------------------------------------------------- image view */
  function borderHit(a, b) {
    // first point where the segment a->b leaves the frame (normalised coordinates)
    var t = 1;
    if (b.u > 1) t = Math.min(t, (1 - a.u) / (b.u - a.u));
    if (b.u < -1) t = Math.min(t, (-1 - a.u) / (b.u - a.u));
    if (b.v > NB) t = Math.min(t, (NB - a.v) / (b.v - a.v));
    if (b.v < -NA) t = Math.min(t, (-NA - a.v) / (b.v - a.v));
    return { u: a.u + (b.u - a.u) * t, v: a.v + (b.v - a.v) * t };
  }
  function drawImage(st) {
    var fr = L.fr, sc = L.sc, zc = st.zc;
    ctx.save(); clipRect(L.img);
    ctx.fillStyle = C.surface; ctx.fillRect(L.img.x, L.img.y, L.img.w, L.img.h);
    // outside the frame: fine hatching
    ctx.save();
    ctx.beginPath(); ctx.rect(L.img.x, L.img.y, L.img.w, L.img.h); ctx.rect(fr.x, fr.y, fr.w, fr.h); ctx.clip("evenodd");
    ctx.strokeStyle = C.soft; ctx.lineWidth = 1;
    var step = 9 * sc;
    ctx.beginPath();
    for (var d = -L.img.h; d < L.img.w; d += step) { ctx.moveTo(L.img.x + d, L.img.y + L.img.h); ctx.lineTo(L.img.x + d + L.img.h, L.img.y); }
    ctx.stroke();
    ctx.restore();
    // the scene inside the frame
    ctx.save(); clipRect(fr);
    ctx.fillStyle = C.bg; ctx.fillRect(fr.x, fr.y, fr.w, fr.h);
    // ground below the horizon, very light
    var hy = fr.y + NA / (NA + NB) * fr.h;
    ctx.fillStyle = C.sunken; ctx.globalAlpha = 0.55; ctx.fillRect(fr.x, hy, fr.w, fr.y + fr.h - hy); ctx.globalAlpha = 1;
    ctx.strokeStyle = C.ink3; ctx.lineCap = "round";
    lines.forEach(function (ln) {
      ctx.lineWidth = ln.w * sc;
      ctx.beginPath();
      var pen = false;
      for (var j = 0; j < ln.p.length; j++) {
        var q = proj({ x: ln.p[j][0], y: ln.p[j][1], z: ln.p[j][2] }, zc);
        if (!q || q.Z < 0.6) { pen = false; continue; }
        var px = imgPx(q);
        if (!pen) { ctx.moveTo(px[0], px[1]); pen = true; } else ctx.lineTo(px[0], px[1]);
      }
      ctx.stroke();
    });
    // scene points (all of them, at the current camera position)
    ctx.fillStyle = C.ink2;
    pts.forEach(function (p) {
      var q = proj(p, zc);
      if (!q || q.Z < 0.6 || p.isX) return;
      var px = imgPx(q), r = clamp(2.6 / Math.sqrt(q.Z / 6), 1, 3.2) * sc;
      ctx.beginPath(); ctx.arc(px[0], px[1], r, 0, Math.PI * 2); ctx.fill();
    });
    // phase 1: matches inside frame t+1 (blue)
    withAlpha(st.a[1], function () {
      tracked.forEach(function (tr) {
        if (tr.oof || !tr.show || !tr.n) return;
        var a = imgPx(tr.t), b = imgPx(tr.n);
        arrow(a[0], a[1], b[0], b[1], C.net, 1.5 * sc, 6 * sc);
      });
    });
    ctx.restore();   // end of frame clip

    // phase 2: out-of-frame points (orange), their match cut at the border
    withAlpha(st.a[2], function () {
      tracked.forEach(function (tr) {
        if (!tr.oof) return;
        var a = imgPx(tr.t);
        withAlpha(1 - 0.75 * st.a[4], function () {
          if (tr.n) {
            var hb = imgPx(borderHit(tr.t, tr.n));
            ctx.strokeStyle = C.oof; ctx.lineWidth = 1.6 * sc; ctx.setLineDash([3 * sc, 3 * sc]);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(hb[0], hb[1]); ctx.stroke(); ctx.setLineDash([]);
            cross(hb[0], hb[1], 3.4 * sc, C.oof);
          }
        });
        ctx.fillStyle = C.oof; ctx.beginPath(); ctx.arc(a[0], a[1], 3.6 * sc, 0, Math.PI * 2); ctx.fill();
      });
    });
    // phase 3: backward matches into frame t-1 (green, short)
    withAlpha(st.a[3] * (1 - 0.55 * st.a[4]), function () {
      tracked.forEach(function (tr) {
        if (!tr.oof || !tr.prev) return;
        var a = imgPx(tr.t), b = imgPx(tr.b);
        var e = [a[0] + (b[0] - a[0]) * st.back, a[1] + (b[1] - a[1]) * st.back];
        arrow(a[0], a[1], e[0], e[1], C.fill, 1.6 * sc, 5.5 * sc);
      });
    });
    // phase 4: reprojected rigid flow, through the border (green)
    withAlpha(st.a[4], function () {
      tracked.forEach(function (tr) {
        if (!tr.oof || !tr.prev || !tr.n) return;
        var a = imgPx(tr.t), b = imgPx(tr.n);
        var e = [a[0] + (b[0] - a[0]) * st.grow, a[1] + (b[1] - a[1]) * st.grow];
        arrow(a[0], a[1], e[0], e[1], C.fill, 2 * sc, 7 * sc);
        if (st.grow > 0.97) { ctx.strokeStyle = C.fill; ctx.lineWidth = 1.4 * sc; ctx.beginPath(); ctx.arc(b[0], b[1], 4.4 * sc, 0, Math.PI * 2); ctx.stroke(); }
      });
    });
    // the frame itself
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2 * sc; ctx.strokeRect(fr.x, fr.y, fr.w, fr.h);
    // X in the image
    withAlpha(st.a[1], function () {
      var tx = tracked.filter(function (tr) { return tr.p.isX; })[0];
      if (!tx) return;
      var a = imgPx(tx.t);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4 * sc; ctx.beginPath(); ctx.arc(a[0], a[1], 9 * sc, 0, Math.PI * 2); ctx.stroke();
      ctx.font = "italic 600 " + fpx(16) + "px 'Source Serif 4', Georgia, serif";
      ctx.fillStyle = C.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText("X", a[0] - 12 * sc, a[1] - 10 * sc);
    });
    // labels
    var ly = fr.y - 14 * sc;
    ctx.font = "600 " + fpx(14) + "px Barlow, Arial, sans-serif";
    ctx.fillStyle = C.ink2; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    var lead = "Image of camera ";
    ctx.fillText(lead, fr.x, ly);
    italicLabel("t", "", fr.x + ctx.measureText(lead).width, ly, C.ink2, 14, "left");
    label("outside the image", L.img.x + 16 * sc, L.img.y + 18 * sc, C.ink3, 13, "left", 500);
    // phase labels: above the frame on the right (wide), centred below it (portrait)
    var px_ = L.portrait ? fr.x + fr.w / 2 : fr.x + fr.w, py_ = L.portrait ? L.img.y + L.img.h - 16 * sc : ly;
    var al = L.portrait ? "center" : "right";
    withAlpha(st.a[2] * (1 - st.a[3]), function () { label("no pixel to compare with", px_, py_, C.oofText, 14, al, 600); });
    withAlpha(st.a[3] * (1 - st.a[4]), function () { label("visible in frame t−1, so depth is known", px_, py_, C.fill, 14, al, 600); });
    withAlpha(st.a[4], function () { label("rigid flow continues past the border", px_, py_, C.fill, 14, al, 600); });
    ctx.restore();
  }

  /* ---------------------------------------------------------------- loop */
  var T = 0, last = 0, playing = !mqReduce.matches, visible = true, raf = 0, ready = false;
  var nowBox = fig.querySelector("[data-now]"), nowPhase = -1;
  var forcedEnd = mqReduce.matches;          // reduced motion: show the final frame of each phase

  function render() {
    if (!ready) return;
    var st = stateAt(T);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, L.w, L.h);
    ctx.globalAlpha = 1;
    drawTop(st);
    ctx.save(); ctx.globalAlpha = 1;
    drawImage(st);
    ctx.restore();
    if (st.g < 1) { ctx.fillStyle = C.surface; ctx.globalAlpha = 1 - st.g; ctx.fillRect(0, 0, L.w, L.h); ctx.globalAlpha = 1; }
    if (nowBox && nowPhase !== st.phase) { nowPhase = st.phase; nowBox.innerHTML = stepBtns[st.phase].querySelector(".explainer__txt").innerHTML; }
    stepBtns.forEach(function (b, idx) {
      var on = idx === st.phase;
      b.setAttribute("aria-pressed", on ? "true" : "false");
      b.parentNode.classList.toggle("is-current", on);
      b.style.setProperty("--p", (on ? (playing ? st.prog * 100 : 100) : (idx < st.phase ? 100 : 0)) + "%");
    });
  }
  function tick(now) {
    raf = 0;
    if (!playing || !visible) return;
    var dt = last ? Math.min((now - last) / 1000, 0.05) : 0;
    last = now;
    T += dt;
    if (T >= TOTAL) T = 0;
    render();
    raf = requestAnimationFrame(tick);
  }
  function start() { if (!raf && playing && visible) { last = 0; raf = requestAnimationFrame(tick); } }
  function setPlaying(p) {
    playing = p;
    playBtn.classList.toggle("is-paused", !p);
    playBtn.setAttribute("aria-label", p ? "Pause animation" : "Play animation");
    if (p) start(); else { if (raf) cancelAnimationFrame(raf); raf = 0; render(); }
  }
  function phaseEnd(p) { return p === DUR.length - 1 ? START[DUR.length] + HOLD * 0.5 : START[p + 1] - 0.001; }

  playBtn.addEventListener("click", function () {
    if (!playing && T >= START[DUR.length]) T = 0;
    setPlaying(!playing);
  });
  stepBtns.forEach(function (b) {
    b.addEventListener("click", function () {
      var p = +b.getAttribute("data-phase");
      if (playing) { T = START[p]; render(); } else { T = phaseEnd(p); render(); }
    });
  });

  function resize() { if (layout()) { ready = true; render(); } }
  if ("ResizeObserver" in window) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener("resize", resize);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (es) {
      visible = es[0].isIntersecting;
      if (visible) start();
    }, { threshold: 0.05 }).observe(canvas);
  }
  document.addEventListener("visibilitychange", function () {
    visible = !document.hidden;
    if (visible) start();
  });
  function onTheme() { readColours(); render(); }
  new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", onTheme);
  mqReduce.addEventListener("change", function () {
    forcedEnd = mqReduce.matches;
    if (forcedEnd) { T = phaseEnd(DUR.length - 1); setPlaying(false); }
  });

  readColours();
  if (forcedEnd) { T = phaseEnd(DUR.length - 1); playing = false; playBtn.classList.add("is-paused"); playBtn.setAttribute("aria-label", "Play animation"); }
  resize();
  start();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { render(); });
  window.__explainer = { tracked: tracked.length, oof: tracked.filter(function (t) { return t.oof; }).length };
})();
