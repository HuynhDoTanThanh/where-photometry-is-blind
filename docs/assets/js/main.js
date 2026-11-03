/* Page behaviour: theme, section highlighting, data reveals, the method walk-through, the
   before/after slider and copy buttons. No dependencies. Every animation has a static end state
   that is shown at once under prefers-reduced-motion or without IntersectionObserver. */
(function () {
  "use strict";
  var root = document.documentElement;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var hasIO = "IntersectionObserver" in window;
  if (!reduce && hasIO) root.classList.add("js-anim");

  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $$(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }

  /* ------------------------------------------------------------ theme */
  var mqDark = window.matchMedia("(prefers-color-scheme: dark)");
  var toggle = $("[data-theme-toggle]");
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    return t === "light" || t === "dark" ? t : (mqDark.matches ? "dark" : "light");
  }
  function syncToggle() {
    if (toggle) toggle.setAttribute("aria-label", effectiveTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store("rf-theme", next);
      syncToggle();
    });
  }
  mqDark.addEventListener("change", syncToggle);
  syncToggle();

  /* ------------------------------------------------------------ nav: current section */
  var links = $$(".nav__links a");
  if (hasIO && links.length) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var inView = {};
    var navIO = new IntersectionObserver(function (es) {
      es.forEach(function (e) { inView[e.target.id] = e.isIntersecting; });
      var cur = null;
      Object.keys(byId).forEach(function (id) { if (inView[id] && !cur) cur = id; });
      links.forEach(function (a) { if (byId[cur] === a) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
    }, { rootMargin: "-40% 0px -55% 0px" });
    Object.keys(byId).forEach(function (id) { var s = document.getElementById(id); if (s) navIO.observe(s); });
  }

  /* ------------------------------------------------------------ one-shot reveals */
  function onReveal(el, fn, threshold) {
    if (reduce || !hasIO) { fn(true); return; }
    var io = new IntersectionObserver(function (es) {
      if (es[0].isIntersecting) { io.disconnect(); fn(false); }
    }, { threshold: threshold || 0.35 });
    io.observe(el);
  }
  $$("[data-reveal]").forEach(function (el) { onReveal(el, function () { el.classList.add("is-in"); }); });

  /* ------------------------------------------------------------ blind-spot mosaics */
  var COLS = 40, ROWS = 25, CELL = 10, GAP = 2;
  function borderOrder() {
    // Cells ordered by distance to the left, right or bottom border (where pixels leave a
    // forward-moving camera's image), bottom rows first within a ring.
    var cells = [];
    for (var r = 0; r < ROWS; r++) for (var c = 0; c < COLS; c++) {
      var d = Math.min(c, COLS - 1 - c, ROWS - 1 - r);
      cells.push({ r: r, c: c, d: d });
    }
    cells.sort(function (a, b) { return a.d - b.d || b.r - a.r || Math.abs(b.c - 19.5) - Math.abs(a.c - 19.5); });
    return cells;
  }
  var order = borderOrder();
  var mosaics = $$("[data-mosaic]").map(function (svg) {
    var solid = +svg.getAttribute("data-solid"), range = +svg.getAttribute("data-range");
    var NS = "http://www.w3.org/2000/svg";
    var rects = [];
    order.forEach(function (cell, idx) {
      var rc = document.createElementNS(NS, "rect");
      rc.setAttribute("x", cell.c * (CELL + GAP));
      rc.setAttribute("y", cell.r * (CELL + GAP));
      rc.setAttribute("width", CELL); rc.setAttribute("height", CELL); rc.setAttribute("rx", 1.5);
      svg.appendChild(rc);
      rects[idx] = rc;
    });
    var fig = svg.closest("figure");
    return { solid: solid, range: range, rects: rects, num: $("[data-count]", fig), rangeText: $("[data-range-text]", fig) };
  });
  function paintMosaic(m, frac) {
    var n = Math.round(m.solid * frac);
    for (var i = 0; i < m.rects.length; i++) {
      var on = i < n, rg = !on && frac >= 1 && i < m.solid + m.range;
      m.rects[i].setAttribute("class", on ? "on" : rg ? "range" : "");
    }
    var target = parseFloat(m.num.getAttribute("data-count"));
    m.num.textContent = (target * frac).toFixed(1);
    if (m.rangeText) m.rangeText.classList.toggle("is-hidden", frac < 1);
  }
  var mosaicWrap = $("[data-mosaics]");
  if (mosaicWrap && mosaics.length) {
    var animate = !reduce && hasIO;
    mosaics.forEach(function (m) { paintMosaic(m, animate ? 0 : 1); });
    if (animate) {
      onReveal(mosaicWrap, function () {
        var t0 = 0, dur = 1500;
        function step(now) {
          if (!t0) t0 = now;
          var f = Math.min((now - t0) / dur, 1);
          var e = 1 - Math.pow(1 - f, 2.2);
          mosaics.forEach(function (m) { paintMosaic(m, f >= 1 ? 1 : e); });
          if (f < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
      }, 0.3);
    }
  }

  /* ------------------------------------------------------------ method walk-through */
  var walk = $("[data-walk]");
  if (walk) {
    var steps = $$("[data-step]", walk);
    var stage = $("[data-stage]", walk);
    var layers = $$(".stage__layer", stage);
    var legend = $("[data-legend]", walk);
    var outToggle = $("[data-out-toggle]", walk);
    var dotsBox = $("[data-walk-dots]", walk);
    var prev = $("[data-walk-prev]", walk), next = $("[data-walk-next]", walk);
    var segG = $('[data-overlay="segments"]', stage), arrG = $('[data-overlay="arrows"]', stage);
    var NS = "http://www.w3.org/2000/svg";
    var cur = -1, outTimer = 0, arrowRaf = 0, overlay = null;
    var desktop = window.matchMedia("(min-width: 960px)");

    var LEGEND = [
      "Frame <i>t</i> of KITTI-12 image 000034, right part.",
      '<span class="k"><b class="sw" style="background:#1F5F95"></b>reliable forward correspondences</span>',
      '<span class="k"><b class="sw" style="background:linear-gradient(90deg,#E6F0F7,#2F6F9F,#0E2236);width:44px"></b>near to far</span><span class="k"><b class="sw" style="background:#CDD1D4"></b>no depth</span>',
      '<span class="k"><b class="sw" style="background:var(--oof-mark)"></b>rigid-flow vectors that end outside the image</span>',
      '<span class="k"><b class="sw" style="background:#2F7A3A;opacity:.6"></b>pixels where the gate switches to <i>T</i></span>',
      '<span class="k"><b class="sw" style="background:#C81E28"></b>outliers</span><span class="k"><b class="sw" style="background:#1C5C28;height:2px"></b>replaced region</span>'
    ];

    var dots = steps.map(function (st, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = st.getAttribute("data-dot");
      b.setAttribute("aria-label", "Step: " + st.querySelector("h3").textContent.replace(/^\s*(In|Out|\d)/, "").trim());
      b.addEventListener("click", function () { go(i, true); });
      dotsBox.appendChild(b);
      return b;
    });

    function loadOverlay(cb) {
      if (overlay) { cb(); return; }
      fetch("assets/img/m_overlay.json").then(function (r) { return r.json(); }).then(function (j) {
        overlay = j;
        j.segments.forEach(function (sg, i) {
          var ln = document.createElementNS(NS, "line");
          ln.setAttribute("x1", sg[0]); ln.setAttribute("y1", sg[1]); ln.setAttribute("x2", sg[2]); ln.setAttribute("y2", sg[3]);
          ln.setAttribute("class", "seg"); ln.setAttribute("pathLength", "1");
          ln.style.strokeDasharray = "1"; ln.style.strokeDashoffset = "1";
          ln.style.transition = reduce ? "none" : "stroke-dashoffset 0.5s ease " + (i * 6) + "ms";
          segG.appendChild(ln);
          var c = document.createElementNS(NS, "circle");
          c.setAttribute("cx", sg[0]); c.setAttribute("cy", sg[1]); c.setAttribute("r", 1.8); c.setAttribute("class", "seg-dot");
          segG.appendChild(c);
        });
        j.oof_arrows.forEach(function (a) {
          var ln = document.createElementNS(NS, "line");
          ln.setAttribute("x1", a[0]); ln.setAttribute("y1", a[1]); ln.setAttribute("x2", a[0]); ln.setAttribute("y2", a[1]);
          ln.setAttribute("class", "oof"); ln.setAttribute("marker-end", "url(#ah-oof)");
          ln._to = [a[2], a[3]]; ln._from = [a[0], a[1]];
          arrG.appendChild(ln);
        });
        cb();
      }).catch(function () { cb(); });
    }
    function showSegments(on) {
      segG.style.display = on ? "" : "none";
      $$("line", segG).forEach(function (ln) { ln.style.strokeDashoffset = on ? "0" : "1"; });
    }
    function showArrows(on) {
      if (arrowRaf) cancelAnimationFrame(arrowRaf);
      arrG.style.display = on ? "" : "none";
      var lns = $$("line", arrG);
      function set(f) { lns.forEach(function (ln) {
        ln.setAttribute("x2", ln._from[0] + (ln._to[0] - ln._from[0]) * f);
        ln.setAttribute("y2", ln._from[1] + (ln._to[1] - ln._from[1]) * f);
      }); }
      if (!on) { set(0); return; }
      if (reduce) { set(1); return; }
      var t0 = 0;
      (function grow(now) {
        if (!t0) t0 = now;
        var f = Math.min((now - t0) / 900, 1);
        set(1 - Math.pow(1 - f, 3));
        if (f < 1) arrowRaf = requestAnimationFrame(grow);
      })(performance.now());
    }
    function showLayer(name) { layers.forEach(function (l) { l.classList.toggle("is-on", l.getAttribute("data-layer") === name); }); }
    function setOut(which) {
      showLayer(which);
      $$("button", outToggle).forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-show") === which ? "true" : "false"); });
    }
    $$("button", outToggle).forEach(function (b) {
      b.addEventListener("click", function () { clearTimeout(outTimer); setOut(b.getAttribute("data-show")); });
    });

    function activate(i) {
      if (i === cur) return;
      cur = i;
      steps.forEach(function (st, k) { st.classList.toggle("is-active", k === i); });
      dots.forEach(function (d, k) { if (k === i) d.setAttribute("aria-current", "step"); else d.removeAttribute("aria-current"); });
      prev.disabled = i === 0; next.disabled = i === steps.length - 1;
      var st = steps[i];
      stage.setAttribute("data-step", i);
      clearTimeout(outTimer);
      legend.innerHTML = LEGEND[i] || "";
      outToggle.hidden = i !== steps.length - 1;
      var ov = st.getAttribute("data-overlay-show");
      loadOverlay(function () {
        if (cur !== i) return;
        showSegments(ov === "segments");
        showArrows(ov === "arrows");
      });
      if (i === steps.length - 1) {
        if (reduce) setOut("out");
        else { setOut("base"); outTimer = setTimeout(function () { setOut("out"); }, 1100); }
      } else showLayer(st.getAttribute("data-layer-show"));
    }
    function go(i, fromUser) {
      i = Math.max(0, Math.min(steps.length - 1, i));
      if (desktop.matches && fromUser) {
        steps[i].scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
        activate(i);
      } else activate(i);
    }
    prev.addEventListener("click", function () { go(cur - 1, true); });
    next.addEventListener("click", function () { go(cur + 1, true); });

    if (hasIO) {
      var stepIO = new IntersectionObserver(function (es) {
        if (!desktop.matches) return;
        es.forEach(function (e) { if (e.isIntersecting) activate(steps.indexOf(e.target)); });
      }, { rootMargin: "-45% 0px -50% 0px" });
      steps.forEach(function (st) { stepIO.observe(st); });
    }
    activate(0);
  }

  /* ------------------------------------------------------------ before/after slider */
  var block = $("[data-compare-block]");
  if (block) {
    var cmp = $("[data-compare]", block);
    var range = $(".compare__range", cmp);
    var before = $("[data-before]", cmp), after = $("[data-after]", cmp), gate = $("[data-gate]", cmp);
    var ctxImg = $("[data-ctx]", block);
    var touched = false;
    function setPos(p) {
      p = Math.max(0, Math.min(100, p));
      cmp.style.setProperty("--pos", p + "%");
      range.value = Math.round(p);
      range.setAttribute("aria-valuetext", Math.round(p) + "% of the width shows U²Flow");
    }
    function fromEvent(e) { var r = cmp.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 100; }
    var dragging = false;
    cmp.addEventListener("pointerdown", function (e) {
      if (e.button !== undefined && e.button !== 0) return;
      dragging = true; touched = true;
      try { cmp.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      setPos(fromEvent(e));
    });
    cmp.addEventListener("pointermove", function (e) { if (dragging) setPos(fromEvent(e)); });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach(function (t) { cmp.addEventListener(t, function () { dragging = false; }); });
    range.addEventListener("input", function () { touched = true; setPos(+range.value); });
    setPos(50);

    // a one-time hint sweep that shows the divider can be moved
    onReveal(cmp, function (instant) {
      if (instant || touched) return;
      var keys = [[0, 50], [650, 74], [1500, 28], [2100, 50]], t0 = 0;
      function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
      (function sweep(now) {
        if (touched) return;
        if (!t0) t0 = now;
        var t = now - t0, k = 1;
        while (k < keys.length - 1 && t > keys[k][0]) k++;
        var a = keys[k - 1], b = keys[k], f = Math.min(Math.max((t - a[0]) / (b[0] - a[0]), 0), 1);
        setPos(a[1] + (b[1] - a[1]) * ease(f));
        if (t < keys[keys.length - 1][0]) requestAnimationFrame(sweep); else setPos(50);
      })(performance.now());
    }, 0.6);

    var tabs = $$('[role="tab"]', block);
    function selectScene(btn) {
      var id = btn.getAttribute("data-scene");
      tabs.forEach(function (t) {
        var on = t === btn;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
      });
      $("#compare-panel").setAttribute("aria-labelledby", btn.id);
      var key = id.slice(-3);
      before.src = "assets/img/k12_" + id + "_u2flow.jpg";
      after.src = "assets/img/k12_" + id + "_rigidfill.jpg";
      gate.src = "assets/img/k12_" + id + "_gate.png";
      ctxImg.src = "assets/img/k12_" + id + "_frame_oof.jpg";
      before.alt = before.getAttribute("data-alt-" + key);
      after.alt = after.getAttribute("data-alt-" + key);
      $$("[data-scene-text]", block).forEach(function (p) { p.hidden = p.getAttribute("data-scene-text") !== id; });
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { selectScene(t); });
      t.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        var n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        n.focus(); selectScene(n);
      });
    });
    var gateBox = $("[data-gate-toggle]", block);
    gateBox.addEventListener("change", function () { cmp.classList.toggle("no-gate", !gateBox.checked); });
    // preload the second scene
    ["u2flow", "rigidfill", "frame_oof"].forEach(function (k) { var im = new Image(); im.src = "assets/img/k12_000140_" + k + ".jpg"; });
  }

  /* ------------------------------------------------------------ copy buttons */
  $$("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var src = btn.parentNode.querySelector("[data-copy-src]");
      var text = src ? src.textContent.trim() : "";
      function done(ok) {
        btn.textContent = ok ? "Copied" : "Press Ctrl+C";
        btn.classList.toggle("is-done", ok);
        setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("is-done"); }, 1800);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { selectText(src); done(false); });
      } else { selectText(src); done(false); }
    });
  });
  function selectText(el) {
    if (!el) return;
    var r = document.createRange(); r.selectNodeContents(el);
    var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  }
})();
