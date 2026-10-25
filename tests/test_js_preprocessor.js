/* Run: PYTHON=/path/to/python node tests/test_js_preprocessor.js
 * Python requires NumPy/SciPy; JS implementation has zero dependencies.
 * All trajectories/scalers below are synthetic; no production parity claimed.
 */
'use strict';
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const path = require('node:path');
const {preprocessV5} = require('../src/export/v5_preprocessor.js');
const root = path.resolve(__dirname, '..');
const python = process.env.PYTHON || 'python3';
const oracle = `
import json, sys
from src.data.preprocessor import V5Preprocessor
cases = json.load(sys.stdin)
result = []
for case in cases:
    p, s = case['input'], case['scaler']
    v = V5Preprocessor(s['mean'], s['scale'], **case.get('options', {}))
    result.append(v.transform(p['x'], p['y'], p['timestamps'], p['pressure']).tolist())
json.dump(result, sys.stdout, allow_nan=False)
`;
const scaler = {feature_names: ['x', 'y', 'pressure', 'velocity', 'dt'],
  mean: [.05, -.1, .2, .6, .03], scale: [.8, .7, 1.2, 2.5, .08]};
function trajectory(n, pressureMode, irregular = false) {
  let t = 0;
  const input = {x: [], y: [], timestamps: [], pressure: pressureMode === 'missing' ? null : []};
  for (let i = 0; i < n; i++) {
    if (i) t += irregular ? .007 + .004 * (1 + Math.sin(i * .7)) : .01;
    input.timestamps.push(t);
    input.x.push(300 + 80 * Math.cos(t * 4.1) + 2 * Math.sin(2 * Math.PI * 25 * t));
    input.y.push(200 + 50 * Math.sin(t * 3.7));
    if (input.pressure) input.pressure.push(pressureMode === 'constant' ? .5 : .5 + .1 * Math.sin(t * 9));
  }
  return input;
}
const cases = [
  {input: trajectory(41, 'missing')}, // 0.4s -> min 48 grid samples
  {input: trajectory(190, 'real')}, // short window, 256 index interpolation
  {input: trajectory(650, 'missing', true)}, // overlapping windows + final tail
  {input: trajectory(510, 'constant')}, // constant pressure -> proxy
  {input: trajectory(400, 'real', true), options: {window: 128, stride: 53}},
  {input: trajectory(320, 'real')}
].map(c => ({...c, scaler}));
cases[5].input.pressure[40] = null; // Missing reading: NumPy converts None to NaN; JS excludes null.
const run = spawnSync(python, ['-c', oracle], {cwd: root, input: JSON.stringify(cases), encoding: 'utf8', maxBuffer: 20 * 1024 * 1024});
assert.equal(run.status, 0, `Python oracle failed; install NumPy/SciPy and set PYTHON.\n${run.stderr}`);
const expected = JSON.parse(run.stdout);
let maxError = 0, windows = 0;
for (let c = 0; c < cases.length; c++) {
  const actual = preprocessV5(cases[c].input, scaler, cases[c].options);
  assert.equal(actual.length, expected[c].length, `case ${c} window count`);
  actual.forEach((tensor, w) => {
    assert(tensor instanceof Float32Array);
    assert.equal(tensor.length, 5 * 256);
    for (let ch = 0; ch < 5; ch++) for (let i = 0; i < 256; i++) {
      const ref = expected[c][w][ch][i];
      const error = Math.abs(tensor[ch * 256 + i] - ref);
      maxError = Math.max(maxError, error);
      assert(error <= 2e-6 + 2e-6 * Math.abs(ref), `case ${c}, window ${w}, channel ${ch}, t ${i}: delta ${error}`);
    }
    windows++;
  });
}
const base = cases[0].input;
assert.throws(() => preprocessV5(base, {...scaler, scale: [1, 1, 0, 1, 1]}));
assert.throws(() => preprocessV5(base, {...scaler, feature_names: ['wrong']}));
assert.throws(() => preprocessV5({...base, timestamps: base.timestamps.map(() => 0)}, scaler));
assert.throws(() => preprocessV5(base, scaler, {stride: 0}));
assert.throws(() => preprocessV5({x: [0], y: [0], timestamps: [0]}, scaler));
console.log(`PASS: ${cases.length} synthetic cases, ${windows} windows, max absolute error ${maxError}; channel-first packing and rejection tests passed.`);
