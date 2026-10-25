/* Zero-dependency source-reference V5 preprocessing, NOT artifact-verified.
 * Inputs: aligned x/y, timestamps in repaired, strictly increasing seconds;
 * optional pressure. Stroke flattening/time-unit detection belong to loaders.
 * Returns one channel-first Float32Array(5*256) per usable overlapping window.
 * Source: src/data/preprocessor.py and original extract_trial_windows.
 * No bounding-box or min-max pressure normalization: neither exists in V5.
 * Browser: window.NeuroTraceV5; Node: require(...). No runtime dependencies.
 */
(function (root) {
  'use strict';
  const FEATURES = ['x', 'y', 'pressure', 'velocity', 'dt'];
  // scipy.signal.butter(4, 12, fs=100, output='sos') and sosfilt_zi(sos).
  const SOS = [
    [0.00891445723946302, 0.01782891447892604, 0.00891445723946302, 1, -0.8931036327062872, 0.22516058868743402],
    [1, 2, 1, 1, -1.1552915050582222, 0.5848302129885163]
  ];
  const ZI = [[0.09847021974311049, -0.015264339845943178], [0.8926153230174261, -0.47744553600594264]];
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  function quantile(a, q) {
    const s = a.slice().sort((x, y) => x - y), k = (s.length - 1) * q, i = Math.floor(k);
    return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (k - i);
  }
  const median = a => quantile(a, 0.5);
  const linspace = (end, n) => Array.from({length: n}, (_, i) => i === n - 1 ? end : i * (end / (n - 1)));
  const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  // NumPy round: ties to even (grid length, pressure variation test).
  function roundEven(x) {
    const f = Math.floor(x), d = x - f;
    return d === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(x);
  }
  function interp(dst, src, values) {
    let j = 0;
    return dst.map(t => {
      while (j + 1 < src.length && src[j + 1] < t) j++;
      if (t <= src[0]) return values[0];
      if (t >= src[src.length - 1]) return values[values.length - 1];
      return values[j] + (values[j + 1] - values[j]) * ((t - src[j]) / (src[j + 1] - src[j]));
    });
  }
  function sosFilter(a, initial) {
    let out = a.slice();
    for (let section = 0; section < SOS.length; section++) {
      const [b0, b1, b2, , a1, a2] = SOS[section];
      let z0 = ZI[section][0] * initial, z1 = ZI[section][1] * initial;
      out = out.map(x => {
        const y = b0 * x + z0;
        z0 = b1 * x - a1 * y + z1;
        z1 = b2 * x - a2 * y;
        return y;
      });
    }
    return out;
  }
  function lowpass(a) {
    // scipy.sosfiltfilt default odd extension; padlen=3*(2*2+1)=15.
    const edge = 15;
    if (a.length <= edge) throw new RangeError('Signal too short for filter');
    const left = a.slice(1, edge + 1).reverse().map(v => 2 * a[0] - v);
    const right = a.slice(-edge - 1, -1).reverse().map(v => 2 * a[a.length - 1] - v);
    const ext = left.concat(a, right);
    const fwd = sosFilter(ext, ext[0]).reverse();
    return sosFilter(fwd, fwd[0]).reverse().slice(edge, -edge);
  }
  function gradient(a) {
    return a.map((_, i) => i === 0 ? (a[1] - a[0]) / 0.01 :
      i === a.length - 1 ? (a[i] - a[i - 1]) / 0.01 : (a[i + 1] - a[i - 1]) / 0.02);
  }
  function preprocessV5(input, scaler, options = {}) {
    const {x, y, timestamps, pressure = null} = input;
    if (!x || !y || !timestamps || x.length < 20 || y.length !== x.length || timestamps.length !== x.length ||
        (pressure !== null && pressure.length !== x.length)) throw new TypeError('Aligned trajectories >=20 points required');
    if (!Array.from(timestamps).every((v, i) => Number.isFinite(v) && (!i || v > timestamps[i - 1])))
      throw new RangeError('Timestamps must be finite, strictly increasing seconds');
    if (!scaler || JSON.stringify(scaler.feature_names) !== JSON.stringify(FEATURES) ||
        scaler.mean?.length !== 5 || scaler.scale?.length !== 5 ||
        !scaler.mean.every(Number.isFinite) || !scaler.scale.every(v => Number.isFinite(v) && v > 0))
      throw new TypeError('V5 scaler requires channel order, five finite means and positive scales');
    const windowSize = options.window ?? 256, stride = options.stride ?? 64;
    if (!Number.isInteger(windowSize) || windowSize < 24 || !Number.isInteger(stride) || stride < 1)
      throw new RangeError('Invalid window or stride');
    const xx = [], yy = [], tt = [], pp = [];
    for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i]) && Number.isFinite(y[i])) {
      xx.push(x[i]); yy.push(y[i]); tt.push(timestamps[i]);
      pp.push(pressure === null ? NaN : pressure[i]);
    }
    if (xx.length < 20) throw new RangeError('Too few finite samples');
    const dt = tt.map((v, i) => i ? v - tt[i - 1] : tt[1] - tt[0]);
    const positive = dt.filter(v => v > 0);
    if (positive.length < 16) throw new RangeError('Insufficient positive intervals');
    const md = median(positive), rep = dt.map(v => v > 0 ? v : md), clean = [0];
    for (let i = 1; i < dt.length; i++) clean.push(clean[i - 1] + clip(rep[i], .05 * md, 10 * md));
    const duration = clean[clean.length - 1];
    if (duration < .4) throw new RangeError('Duration below 0.4 seconds');
    const grid = linspace(duration, Math.max(48, roundEven(duration * 100) + 1));
    const xg = lowpass(interp(grid, clean, xx)), yg = lowpass(interp(grid, clean, yy));
    const dg = lowpass(interp(grid, clean, rep));
    const cx = median(xg), cy = median(yg);
    const radius = xg.map((v, i) => Math.hypot(v - cx, yg[i] - cy));
    let scale = quantile(radius, .95);
    if (!Number.isFinite(scale) || scale <= 1e-9) scale = 1;
    const vx = gradient(xg.map(v => (v - cx) / scale));
    const vy = gradient(yg.map(v => (v - cy) / scale));
    const speed = vx.map((v, i) => Math.hypot(v, vy[i]));
    const finiteP = pp.filter(Number.isFinite);
    let pg;
    if (pressure !== null && finiteP.length && new Set(finiteP.map(v => roundEven(v * 1e4))).size >= 4) {
      const mp = median(finiteP);
      pg = lowpass(interp(grid, clean, pp.map(v => Number.isFinite(v) ? v : mp)));
    } else {
      const lo = speed.reduce((a, b) => Math.min(a, b), Infinity);
      const hi = speed.reduce((a, b) => Math.max(a, b), -Infinity);
      pg = speed.map((v, i) => .65 * (1 - .35 * ((v - lo) / (hi - lo + 1e-6))) + .15 * clip(radius[i] / scale, 0, 1.5));
    }
    const n = grid.length, starts = [];
    if (n <= windowSize) starts.push(0);
    else {
      for (let s = 0; s <= n - windowSize; s += stride) starts.push(s);
      if (starts[starts.length - 1] !== n - windowSize) starts.push(n - windowSize);
    }
    const result = [];
    for (const start of starts) {
      const channels = [xg, yg, pg, speed, dg].map(a => {
        const w = a.slice(start, start + windowSize);
        return interp(linspace(1, 256), linspace(1, w.length), w);
      });
      const mx = mean(channels[0]), my = mean(channels[1]);
      const wx = channels[0].map(v => v - mx), wy = channels[1].map(v => v - my);
      const ws = Math.sqrt(mean(wx.map((v, i) => v * v + wy[i] * wy[i])));
      if (ws <= 1e-8) continue;
      const mp = mean(channels[2]), sd = Math.sqrt(mean(channels[2].map(v => (v - mp) ** 2)));
      const medDt = median(channels[4]);
      const normalized = [wx.map(v => v / ws), wy.map(v => v / ws),
        channels[2].map(v => (v - mp) / (sd + 1e-5)), channels[3],
        channels[4].map(v => (v - medDt) / (medDt + 1e-6))];
      // Original emits float32 BEFORE applying float64 scaler, then float32 again.
      const tensor = new Float32Array(5 * 256);
      for (let c = 0; c < 5; c++) for (let i = 0; i < 256; i++)
        tensor[c * 256 + i] = (Math.fround(normalized[c][i]) - scaler.mean[c]) / scaler.scale[c];
      if (tensor.every(Number.isFinite)) result.push(tensor);
    }
    if (!result.length) throw new RangeError('No usable windows');
    return result;
  }
  const api = Object.freeze({preprocessV5});
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NeuroTraceV5 = api;
})(globalThis);
