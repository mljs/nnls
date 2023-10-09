/*
Can be executed using `node -r esm --inspect-brk benchmark/index.js`
And debug from chrome using `chrome://inspect`
*/
import { readFileSync } from 'fs';
import { Matrix } from 'ml-matrix';
import { fcnnls } from '../lib/index.js';
import { join } from 'path';

/**
const { readFileSync } = require('fs');
const { Matrix } = require('ml-matrix');
const { fcnnls } = require('../lib/index.js');
const { join } = require('path');
 */
// const __dirname = new URL('.', import.meta.url).pathname;

const concentration = readFileSync(
  join(__dirname, '../src/__tests__/data/matrix.txt'),
  'utf-8',
);
let linesA = concentration.split(/[\r\n]+/);
let A = [];
for (let line of linesA) {
  A.push(line.split(',').map((value) => Number(value)));
}
let matrix = new Matrix(A);
matrix = matrix.transpose();

const observation = readFileSync(
  join(__dirname, '../src/__tests__/data/target.txt'),
  'utf-8',
);
let lines = observation.split(/[\r\n]+/);
let b = [];
for (let line of lines) {
  b.push(line.split(',').map((value) => Number(value)));
}
let target = new Matrix(b);
target = target.transpose();

console.profile('start');
console.time('flag');
let result = fcnnls(matrix, target);
console.timeEnd('flag');
console.profileEnd();
