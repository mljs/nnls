# nnls

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Non-Negative Least-Squares (NNLS) algorithm, by Lawson and Hanson. It was mostly done for learning purposes, and will be improved over time.

Currently, it seems to match [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html), at least in some basic tests.

You are welcome to report issues and contribute to this project.

## Installation

`$ npm i nnls`

## Basic Usage

```js
import { nnls } from 'nnls';

const { x, d } = nnls(X, y); // result and dual vectors
```

The code returns also the dual vector.

You can get execution information using the options:

## Example with options

```js
import { nnls } from 'nnls';
import { Matrix } from 'ml-matrix'; //npm i ml-matrix
const X = new Matrix([
  [1, 0],
  [2, 0],
  [3, 0],
  [0, 1],
]);
const Y = Matrix.columnVector([1, 2, 3, 4]);
const solution = Matrix.columnVector([1, 4]);
const result = nnls(X, Y, { info: true });
console.log(result.x.to1DArray(), result.info);
/*
  {
  x: Matrix {
    [
      1.000000
      4
    ]
    rows: 2
    columns: 1
  },
  d: Matrix {
    [
      -3.6e-15
       0
    ]
    rows: 2
    columns: 1
  },
  info: {
    rse: [ 5.477225575051661, 4, 1.0175362097255202e-15 ],
    iterations: 3
  }
}
 * /
```

## [Documentation](https://mljs.github.io/nnls/)

## Misc.

Like other implementations (for example `scipy.optimize.nnls`) it is limited to a single vector $y$, or as it is called in the literature, a single right-hand-side (RHS).

As a minor addition to other implementations, you can pass `{ interceptAtZero:false }` then the result is consistent with $f(0)=C$.

For multiple RHS, you can take a look at [Fast-Combinatorial Non-Negative Least-Squares](https://github.com/mljs/fcnnls)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/nnls.svg
[npm-url]: https://www.npmjs.com/package/nnls
[ci-image]: https://github.com/mljs/nnls/workflows/Node.js%20CI/badge.svg?branch=main
[ci-url]: https://github.com/mljs/nnls/actions?query=workflow%3A%22Node.js+CI%22
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/nnls.svg
[codecov-url]: https://codecov.io/gh/mljs/nnls
[download-image]: https://img.shields.io/npm/dm/nnls.svg
[download-url]: https://www.npmjs.com/package/nnls
