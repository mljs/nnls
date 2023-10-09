/*
  profile it using `node --cpu-prof benchmark/index.mjs`
  for benchmark uncomment the bottom lines.
*/
import { readFileSync } from 'fs';
import { join } from 'path';

import { Matrix } from 'ml-matrix';

import { nnls as fcnnls } from '../lib/index.js';

const __dirname = new URL('.', import.meta.url).pathname;

const concentration = readFileSync(join(__dirname, 'matrix.txt'), 'utf-8');
let linesA = concentration.split(/[\r\n]+/);
let A = [];
for (let line of linesA) {
  A.push(line.split(',').map((value) => Number(value)));
}
let matrix = new Matrix(A);
matrix = matrix.transpose();

const observation = readFileSync(join(__dirname, 'target.txt'), 'utf-8');
let lines = observation.split(/[\r\n]+/);
let b = [];
for (let line of lines) {
  b.push(line.split(',').map((value) => Number(value)));
}
let target = new Matrix(b);
target = target.transpose();

fcnnls(matrix, target);
console.profile('start');
console.time('flag');
const result = fcnnls(matrix, target, { info: true });
console.timeEnd('flag');
console.profileEnd();
console.log(result);
