/* The 3D view of the floor atlas (atlas spec 9, 10).
 *
 * Hand-written WebGL 1, no library, no network. Ported from the author's
 * earlier work (PORTED.md) and fed only by the page. It reads exactly two things:
 *
 *   window.__ATLAS__  the payload page.py writes (atlas plan, interface
 *                     contract): one base64 byte buffer and its parts, the
 *                     kinds with their colours, the storeys, the rooms and
 *                     the layers with their legends;
 *   #atlas-3d         the element to draw into. The page gives it its size;
 *                     this script adds the canvas and the controls inside it.
 *
 * Every colour, legend, storey, grid spacing and work-plane height comes from
 * the payload; nothing about a project is written here.
 */
(function () {
  "use strict";
  var D = window.__ATLAS__;
  var slot = document.getElementById("atlas-3d");
  if (!D || !slot) return;

  function note(text) {
    var p = document.createElement("p");
    p.className = "v3d-none";
    p.textContent = text;
    slot.appendChild(p);
  }

  /* ---------------------------------------------------------------- data */
  function bytes(b64) {
    var s = atob(b64), a = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
    return a;
  }
  var RAW = bytes(D.blob).buffer;
  function part(name, Type) {
    var s = D.parts[name];
    return s ? new Type(RAW, s[0], s[1]) : new Type(0);
  }
  /* Positions travel as int16 with one offset and scale; this puts them back. */
  var Q = D.quant;                                   // [ox, oy, oz, scale]
  function dequant(i16) {
    var f = new Float32Array(i16.length);
    for (var i = 0; i < i16.length; i += 3) {
      f[i] = i16[i] * Q[3] + Q[0];
      f[i + 1] = i16[i + 1] * Q[3] + Q[1];
      f[i + 2] = i16[i + 2] * Q[3] + Q[2];
    }
    return f;
  }

  /* Storeys that share a level (a split plan) show together: GROUP[i] is the
     first storey at storey i's level. Site (-1) and below (-2) stay as they are. */
  var STOREYS = D.storeys;
  var GROUP = STOREYS.map(function (s, i) {
    for (var j = 0; j < i; j++) if (STOREYS[j].level === s.level) return j;
    return i;
  });
  var GROUPS = [];
  STOREYS.forEach(function (s, i) {
    if (GROUP[i] === i) GROUPS.push({ index: i, label: s.label });
    else GROUPS.forEach(function (g) { if (g.index === GROUP[i]) g.label += " / " + s.label; });
  });
  function grouped(sto) {
    var f = new Float32Array(sto.length);
    for (var i = 0; i < sto.length; i++) f[i] = sto[i] >= 0 ? GROUP[sto[i]] : sto[i];
    return f;
  }
  function groupLabel(g) {
    for (var i = 0; i < GROUPS.length; i++) if (GROUPS[i].index === g) return GROUPS[i].label;
    return "";
  }

  /* ------------------------------------------------------------- webgl */
  var canvas = document.createElement("canvas");
  var gl = null;
  try { gl = canvas.getContext("webgl", { antialias: true, alpha: false }); } catch (e) { gl = null; }
  if (!gl) {
    note("This browser has no WebGL, so the 3D view cannot be drawn; the storey plans are not affected.");
    return;
  }

  function shader(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function program(vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  /* Storey filtering happens in the fragment shader: one draw call per kind,
     whatever is isolated. uMode 0 draws every storey, 1 only the selected
     one, 2 only the rest. Two passes with 1 then 2 give a highlight; pass 1
     alone gives a solo. */
  var COMMON =
    "uniform float uStorey;uniform float uMode;varying float vS;" +
    "bool cut(){if(uStorey<-0.5)return false;" +
    "bool mine=abs(vS-uStorey)<0.5;" +
    "if(uMode>1.5)return mine;if(uMode>0.5)return !mine;return false;}";

  var progShell, progTile;
  try {
    progShell = program(
      "attribute vec3 aPos;attribute vec3 aNrm;attribute float aS;" +
      "uniform mat4 uMVP;varying float vS;varying float vL;" +
      "void main(){vS=aS;" +
      "vec3 n=normalize(aNrm);vL=0.42+0.58*abs(dot(n,normalize(vec3(0.35,0.5,0.8))));" +
      "gl_Position=uMVP*vec4(aPos,1.0);}",
      "precision mediump float;uniform vec4 uColor;" + COMMON + "varying float vL;" +
      "void main(){if(cut())discard;gl_FragColor=vec4(uColor.rgb*vL,uColor.a);}");
    progTile = program(
      "attribute vec3 aPos;attribute vec3 aCol;attribute float aS;" +
      "uniform mat4 uMVP;varying vec3 vC;varying float vS;" +
      "void main(){vC=aCol;vS=aS;gl_Position=uMVP*vec4(aPos,1.0);}",
      "precision mediump float;varying vec3 vC;" + COMMON +
      "void main(){if(cut())discard;" +
      "if(uMode>1.5){gl_FragColor=vec4(mix(vC,vec3(0.62),0.86),0.30);return;}" +
      "gl_FragColor=vec4(vC,1.0);}");
  } catch (e) {
    note("The 3D view could not start in this browser (" + e.message + "); the storey plans are not affected.");
    return;
  }

  function buffer(data) {
    var b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }

  /* The extent of everything drawn, for the home view. */
  var LO = [Infinity, Infinity, Infinity], HI = [-Infinity, -Infinity, -Infinity];
  function grow(xyz) {
    for (var i = 0; i < xyz.length; i += 3) {
      for (var k = 0; k < 3; k++) {
        if (xyz[i + k] < LO[k]) LO[k] = xyz[i + k];
        if (xyz[i + k] > HI[k]) HI[k] = xyz[i + k];
      }
    }
  }

  /* ------------------------------------------------------------- shell */
  var SHELL = D.kinds.map(function (k) {
    var pos = dequant(part("k_" + k.k + "_pos", Int16Array));
    grow(pos);
    return {
      k: k.k, on: !!k.on,
      rgba: [k.rgba[0] / 255, k.rgba[1] / 255, k.rgba[2] / 255, k.rgba[3] / 255],
      pos: buffer(pos),
      nrm: buffer(part("k_" + k.k + "_nrm", Int8Array)),
      st: buffer(grouped(part("k_" + k.k + "_sto", Int8Array))),
      count: pos.length / 3
    };
  });

  /* ------------------------------------------------------------ layers */
  /* Layer 0 is the model grid; each layer names its parts by `part`. The
     tiles (two triangles per point) are built here on first use. */
  var LAYERS = D.layers.map(function (L) {
    var xyz = dequant(part(L.part + "_pos", Int16Array));
    grow(xyz);
    return {
      L: L, xyz: xyz, n: xyz.length / 3,
      sto: part(L.part + "_sto", Int8Array),
      room: part(L.part + "_room", Uint16Array),
      val: part(L.part + "_val", Float32Array),
      rgb: part(L.part + "_rgb", Uint8Array),
      tiles: null, dots: null, col: null
    };
  });
  if (!isFinite(LO[0])) { LO = [0, 0, 0]; HI = [1, 1, 1]; }
  var C = [(LO[0] + HI[0]) / 2, (LO[1] + HI[1]) / 2, (LO[2] + HI[2]) / 2];
  var R = Math.max(1, Math.hypot(HI[0] - LO[0], HI[1] - LO[1], HI[2] - LO[2]) / 2);

  function tileGeometry(layer, size, lift) {
    var n = layer.n, p = new Float32Array(n * 18), s = new Float32Array(n * 6), h = size / 2;
    var dx = [-h, h, h, -h, h, -h], dy = [-h, -h, h, -h, h, h];
    for (var i = 0; i < n; i++) {
      var g = layer.sto[i] >= 0 ? GROUP[layer.sto[i]] : layer.sto[i];
      for (var k = 0; k < 6; k++) {
        var o = (i * 6 + k) * 3;
        p[o] = layer.xyz[i * 3] + dx[k];
        p[o + 1] = layer.xyz[i * 3 + 1] + dy[k];
        p[o + 2] = layer.xyz[i * 3 + 2] + lift;
        s[i * 6 + k] = g;
      }
    }
    return { pos: buffer(p), st: buffer(s), count: n * 6 };
  }
  function tileColours(layer) {
    var n = layer.n, c = new Float32Array(n * 18);
    for (var i = 0; i < n; i++) {
      var r = layer.rgb[i * 3] / 255, g = layer.rgb[i * 3 + 1] / 255, b = layer.rgb[i * 3 + 2] / 255;
      for (var k = 0; k < 6; k++) {
        var o = (i * 6 + k) * 3; c[o] = r; c[o + 1] = g; c[o + 2] = b;
      }
    }
    return buffer(c);
  }
  function ready(layer) {
    if (!layer.tiles) {
      layer.tiles = tileGeometry(layer, D.cell, 0.0);
      layer.dots = tileGeometry(layer, D.cell * 0.3, 0.02);
      layer.col = tileColours(layer);
    }
    return layer;
  }

  /* ---------------------------------------------------------- matrices */
  function perspective(fov, a, n, f) {
    var t = 1 / Math.tan(fov / 2);
    return [t / a, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0];
  }
  function lookAt(e, c, u) {
    var z = norm(sub(e, c)), x = norm(cross(u, z)), y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, e), -dot(y, e), -dot(z, e), 1];
  }
  function mul(a, b) {
    var o = new Array(16);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
      o[i * 4 + j] = a[j] * b[i * 4] + a[4 + j] * b[i * 4 + 1] +
        a[8 + j] * b[i * 4 + 2] + a[12 + j] * b[i * 4 + 3];
    }
    return o;
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

  /* ------------------------------------------------------------- state */
  var view = {
    yaw: -0.85, pitch: 0.62, dist: R * 2.4, target: C.slice(),
    layer: 0, dots: false, storey: -1, solo: false, on: {}
  };
  SHELL.forEach(function (s) { view.on[s.k] = s.on; });

  /* The extent of one storey's model grid, so a solo storey fills the frame. */
  function storeyFit(g) {
    var layer = LAYERS[0], lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], n = 0;
    for (var i = 0; i < layer.n; i++) {
      if (layer.sto[i] < 0 || GROUP[layer.sto[i]] !== g) continue;
      for (var k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], layer.xyz[i * 3 + k]);
        hi[k] = Math.max(hi[k], layer.xyz[i * 3 + k]);
      }
      n++;
    }
    if (!n) return null;
    return {
      target: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
      radius: Math.max(D.cell * 4, Math.hypot(hi[0] - lo[0], hi[1] - lo[1]) / 2)
    };
  }
  /* A highlight keeps the whole-building frame, so you can see where the
     selected storey sits. Only a solo zooms in. */
  function home() {
    view.yaw = -0.85; view.pitch = 0.62;
    var f = view.storey >= 0 && view.solo ? storeyFit(view.storey) : null;
    if (f) { view.target = f.target; view.dist = f.radius * 2.6; }
    else { view.target = C.slice(); view.dist = R * 2.4; }
  }

  function eye() {
    var cp = Math.cos(view.pitch);
    return [view.target[0] + view.dist * cp * Math.cos(view.yaw),
      view.target[1] + view.dist * cp * Math.sin(view.yaw),
      view.target[2] + view.dist * Math.sin(view.pitch)];
  }
  function mvp(w, h) {
    return mul(perspective(0.85, w / h, Math.max(0.05, R * 0.01), R * 24),
      lookAt(eye(), view.target, [0, 0, 1]));
  }

  /* -------------------------------------------------------------- draw */
  function attrib(p, name, buf, size, type, normed) {
    var loc = gl.getAttribLocation(p, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type || gl.FLOAT, !!normed, 0, 0);
  }

  /* The canvas clears to the page's sheet colour, white when it has none. */
  function sheetColour() {
    var hex = getComputedStyle(slot).getPropertyValue("--sheet").trim();
    var m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return [1, 1, 1];
    var v = parseInt(m[1], 16);
    return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
  }

  function draw() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.round(slot.clientWidth * dpr), h = Math.round(slot.clientHeight * dpr);
    if (!w || !h) return;                            // the tab is hidden
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    var bg = sheetColour();
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    var m = new Float32Array(mvp(w, h));

    var layer = ready(LAYERS[view.layer]);
    if (layer.n) {
      gl.useProgram(progTile);
      gl.uniformMatrix4fv(gl.getUniformLocation(progTile, "uMVP"), false, m);
      gl.uniform1f(gl.getUniformLocation(progTile, "uStorey"), view.storey);
      var uM = gl.getUniformLocation(progTile, "uMode");
      var g = view.dots ? layer.dots : layer.tiles;
      attrib(progTile, "aPos", g.pos, 3);
      attrib(progTile, "aCol", layer.col, 3);
      attrib(progTile, "aS", g.st, 1);
      // the selected storey first, in full colour and writing depth
      gl.uniform1f(uM, view.storey < 0 ? 0 : 1);
      gl.drawArrays(gl.TRIANGLES, 0, g.count);
      // then the rest, washed out, so you can see where the selection sits
      if (view.storey >= 0 && !view.solo) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.uniform1f(uM, 2);
        gl.drawArrays(gl.TRIANGLES, 0, g.count);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }
    }

    gl.useProgram(progShell);
    gl.uniformMatrix4fv(gl.getUniformLocation(progShell, "uMVP"), false, m);
    // the model is cut to the storey only in solo; a highlight keeps the
    // whole building around it
    gl.uniform1f(gl.getUniformLocation(progShell, "uStorey"), view.solo ? view.storey : -1);
    gl.uniform1f(gl.getUniformLocation(progShell, "uMode"), 1);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    SHELL.forEach(function (s) {
      if (!view.on[s.k] || !s.count) return;
      gl.uniform4fv(gl.getUniformLocation(progShell, "uColor"), s.rgba);
      attrib(progShell, "aPos", s.pos, 3);
      attrib(progShell, "aNrm", s.nrm, 3, gl.BYTE, true);
      attrib(progShell, "aS", s.st, 1);
      gl.drawArrays(gl.TRIANGLES, 0, s.count);
    });
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  var pending = false;
  function render() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; draw(); });
  }

  /* ------------------------------------------------------------ picking */
  /* A click becomes a ray; the ray meets the plane of each point of the
     shown layer at that point's own height; the nearest point whose tile
     holds the hit is the one clicked. No render-to-texture needed. */
  function invert(m) {
    var inv = new Array(16), i;
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    var d = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!d) return null;
    for (i = 0; i < 16; i++) inv[i] /= d;
    return inv;
  }
  function unproject(inv, x, y, z) {
    var o = [0, 0, 0, 0];
    for (var i = 0; i < 4; i++) {
      o[i] = inv[i] * x + inv[4 + i] * y + inv[8 + i] * z + inv[12 + i];
    }
    return [o[0] / o[3], o[1] / o[3], o[2] / o[3]];
  }
  function pick(px, py) {
    var r = canvas.getBoundingClientRect();
    var nx = (px - r.left) / r.width * 2 - 1, ny = 1 - (py - r.top) / r.height * 2;
    var inv = invert(mvp(r.width, r.height));
    if (!inv) return null;
    var a = unproject(inv, nx, ny, -1), b = unproject(inv, nx, ny, 1);
    var d = sub(b, a);
    if (Math.abs(d[2]) < 1e-9) return null;
    var layer = LAYERS[view.layer], h = D.cell / 2, best = null, bestT = Infinity;
    for (var i = 0; i < layer.n; i++) {
      var s = layer.sto[i];
      if (view.storey >= 0 && (s < 0 || GROUP[s] !== view.storey)) continue;
      var t = (layer.xyz[i * 3 + 2] - a[2]) / d[2];
      if (t < 0 || t >= bestT) continue;
      var hx = a[0] + d[0] * t, hy = a[1] + d[1] * t;
      if (Math.abs(layer.xyz[i * 3] - hx) <= h && Math.abs(layer.xyz[i * 3 + 1] - hy) <= h) {
        best = i; bestT = t;
      }
    }
    return best;
  }

  /* ---------------------------------------------------------------- ui */
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function fixed(v, n) { return (Math.round(v * Math.pow(10, n)) / Math.pow(10, n)).toFixed(n); }
  function rgbCss(c) { return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")"; }

  function valueText(L, v) {
    if (L.fail_when === null) {
      if (!L.legend.ramp) return L.legend.items[0][0];
      return fixed(v, 1);
    }
    var fail = (v !== 0) === L.fail_when;
    return L.legend.items[fail ? 1 : 0][0];
  }

  var readout = el("div", "v3d-read");
  function say(i) {
    readout.textContent = "";
    if (i == null) {
      readout.appendChild(el("b", null, "Click a tile"));
      readout.appendChild(el("span", "k", "Each tile is one point, " + fixed(D.cell, 2) +
        " m across; the grid lies " + fixed(D.work, 2) + " m above its floor."));
      return;
    }
    var layer = LAYERS[view.layer], L = layer.L;
    var s = layer.sto[i];
    readout.appendChild(el("b", null, D.rooms[layer.room[i]]));
    readout.appendChild(el("span", "k", "storey " + (s >= 0 ? STOREYS[s].label : "")));
    var table = el("table");
    [
      ["x, y, z (m)", fixed(layer.xyz[i * 3], 2) + ", " + fixed(layer.xyz[i * 3 + 1], 2) + ", " +
        fixed(layer.xyz[i * 3 + 2], 2)],
      [L.title, valueText(L, layer.val[i])]
    ].forEach(function (row) {
      var tr = el("tr");
      tr.appendChild(el("td", null, row[0]));
      tr.appendChild(el("td", null, row[1]));
      table.appendChild(tr);
    });
    readout.appendChild(table);
    if (L.legend.ramp) readout.appendChild(el("span", "k", L.unit));
  }

  /* The legend of the shown layer: the ramp with its ticks and marks for a
     number layer, a swatch per label otherwise. All of it from the payload. */
  var legend = el("div", "v3d-scale");
  function drawLegend() {
    var L = LAYERS[view.layer].L, g = L.legend;
    legend.textContent = "";
    legend.appendChild(el("div", "t", L.title));
    if (g.ramp) {
      var bar = el("div", "bar");
      bar.style.background = "linear-gradient(90deg," + g.ramp.map(function (s) {
        return rgbCss(s[1]) + " " + fixed(s[0] * 100, 1) + "%";
      }).join(",") + ")";
      g.marks.forEach(function (t) {
        var m = el("span", "mark");
        m.style.left = fixed(t * 100, 1) + "%";
        bar.appendChild(m);
      });
      legend.appendChild(bar);
      var ticks = el("div", "tk");
      g.ticks.forEach(function (t) {
        var s = el("span", null, t[1]);
        s.style.left = fixed(t[0] * 100, 1) + "%";
        ticks.appendChild(s);
      });
      legend.appendChild(ticks);
      legend.appendChild(el("div", "k", L.unit));
    } else {
      g.items.forEach(function (item) {
        var row = el("div", "item");
        var sw = el("span", "sw");
        sw.style.background = rgbCss(item[1]);
        row.appendChild(sw);
        row.appendChild(el("span", null, item[0]));
        legend.appendChild(row);
      });
    }
  }

  var badge = el("div", "v3d-badge");
  var ui = el("div", "v3d-ui");

  function button(group, value, text) {
    var b = el("button", null, text);
    b.type = "button"; b.dataset.g = group; b.dataset.v = value;
    return b;
  }
  function controls() {
    var layerGroup = el("div", "grp");
    var select = el("select");
    select.setAttribute("aria-label", "layer");
    LAYERS.forEach(function (layer, i) {
      var o = el("option", null, layer.L.title);
      o.value = String(i);
      select.appendChild(o);
    });
    select.addEventListener("change", function () {
      view.layer = parseInt(select.value, 10);
      say(null); sync(); render();
    });
    layerGroup.appendChild(select);
    ui.appendChild(layerGroup);

    var groups = [
      ["marks", [["tiles", "tiles"], ["dots", "dots"]]],
      ["storey", [["-1", "all"]].concat(GROUPS.map(function (g) { return [String(g.index), g.label]; }))],
      ["show", [["solo", "solo"], ["home", "reset view"]]],
      ["kind", SHELL.map(function (s) { return [s.k, s.k.replace(/_/g, " ")]; })]
    ];
    groups.forEach(function (g) {
      var d = el("div", "grp");
      g[1].forEach(function (b) { d.appendChild(button(g[0], b[0], b[1])); });
      ui.appendChild(d);
    });
    ui.addEventListener("click", function (e) {
      var b = e.target.closest("button");
      if (!b) return;
      var g = b.dataset.g, v = b.dataset.v;
      if (g === "marks") view.dots = (v === "dots");
      else if (g === "storey") { view.storey = parseInt(v, 10); if (view.solo) home(); say(null); }
      else if (g === "kind") view.on[v] = !view.on[v];
      else if (v === "solo") { view.solo = !view.solo; home(); }
      else if (v === "home") home();
      sync(); render();
    });
    slot.appendChild(ui);
    slot.appendChild(readout);
    slot.appendChild(legend);
    slot.appendChild(badge);
    slot.appendChild(el("div", "v3d-hint", "drag to turn · wheel to zoom · shift-drag to pan · click a tile"));
  }

  function sync() {
    ui.querySelectorAll("button").forEach(function (b) {
      var g = b.dataset.g, v = b.dataset.v, on = false;
      if (g === "marks") on = view.dots === (v === "dots");
      else if (g === "storey") on = view.storey === parseInt(v, 10);
      else if (g === "kind") on = !!view.on[v];
      else if (v === "solo") on = view.solo;
      b.setAttribute("aria-pressed", String(on));
    });
    drawLegend();
    badge.textContent = (view.storey < 0 ? "All storeys" : "Storey " + groupLabel(view.storey)) +
      " · " + LAYERS[view.layer].L.title;
  }

  /* ---------------------------------------------------------- movement */
  var drag = null;
  canvas.addEventListener("pointerdown", function (e) {
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2, moved: 0 };
  });
  canvas.addEventListener("pointermove", function (e) {
    if (!drag) return;
    var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.pan) {
      var s = view.dist * 1.6e-3;
      var right = [-Math.sin(view.yaw), Math.cos(view.yaw), 0];
      view.target[0] += (-dx * right[0]) * s;
      view.target[1] += (-dx * right[1]) * s;
      view.target[2] += dy * s;
    } else {
      view.yaw -= dx * 0.008;
      view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + dy * 0.008));
    }
    render();
  });
  canvas.addEventListener("pointerup", function (e) {
    if (drag && drag.moved < 5) say(pick(e.clientX, e.clientY));
    drag = null;
  });
  canvas.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  canvas.addEventListener("wheel", function (e) {
    e.preventDefault();
    view.dist = Math.max(R * 0.05, Math.min(R * 9, view.dist * Math.exp(e.deltaY * 1.1e-3)));
    render();
  }, { passive: false });

  /* ------------------------------------------------------------- mount */
  slot.appendChild(canvas);
  controls();
  home();
  say(null);
  sync();
  render();
  window.addEventListener("resize", render);
  /* A hidden tab has no size; draw again when the page shows it. */
  if (window.ResizeObserver) new ResizeObserver(render).observe(slot);
})();
