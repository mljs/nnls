import { data } from './data1.mjs';
import { nnls } from '../lib/index.js';

const X = data.mC;
const y = data.bf;

const init = performance.now();
for (let i = 0; i < 100; i++) {
  nnls(X, y);
}
const end = performance.now();
console.log(end - init);
