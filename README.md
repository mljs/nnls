# nnls

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

This is a pure Typescript implementation of the Non-Negative Least-Squares (NNLS) algorithm, by Lawson and Hanson. It was mostly done for learning purposes, and will be improved over time.

Find details in the [Misc.](#Misc.) section below.

## Installation

`$ npm i nnls`

## Usage

```js
import { nnls } from 'nnls';

// you may need `resultVector`
const { resultVector, dualVector, residualVector } = nnls(X, y);
```

But it is limited to a single vector $y$, or as it is called in the literature, a single right hand side.

For a more faster version, especially if you need multiple right hand sides, you can take a look at [Fast-Combinatorial Non-Negative Least-Squares](https://github.com/mljs/fcnnls)

## Misc.

<details>

<summary>Expand</summary>

In the book [Solving Least Squares Problems](https://books.google.co.uk/books?hl=en&lr=&id=AEwDbHp50FgC&oi=fnd&pg=PP1&ots=dITQ_G2Hcz&sig=-mmloZHdwjlqlEOFrP2azmfer6g#v=onepage&q&f=false) by Lawson & Hanson, 1995, they describe Non-Negative Least-Squares method for solving linear least squares problems with non-negativity constraints (Chapter 23, Section 3.)

Both Least Squares and Non-Negative Least Squares (NNLS) are methods for solving linear regression problems. They become useful when the equation $\mathbf{A}x=b$ has no solution, because $b$ is not in the column space of a $C(A)$.

We can approximate a solution, which is equivalent to finding a vector $x$ in the column space of $A$ that closest to $b$, where closest implies the minimum of the Euclidean norm $||Ax-b||$.

The NNLS problem can be stated as:
$$ \mathrm{argmin}{\_x} ||Ax - b|| \hspace{1em} \mathrm{s.t} \hspace{1em} x \geq 0$$

where $A$ is a matrix of inputs, and $b$ is a vector of outputs.

In linear regression by least squares the analytical solution is $$x = (A^TA)^{-1}A^Tb$$ But NNLS is here solved iteratively, and the solution is based on the active set method.

</details>

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/nnls.svg
[npm-url]: https://www.npmjs.com/package/nnls
[ci-image]: https://github.com/santimirandarp/nnls/workflows/Node.js%20CI/badge.svg?branch=main
[ci-url]: https://github.com/santimirandarp/nnls/actions?query=workflow%3A%22Node.js+CI%22
[codecov-image]: https://img.shields.io/codecov/c/github/santimirandarp/nnls.svg
[codecov-url]: https://codecov.io/gh/santimirandarp/nnls
[download-image]: https://img.shields.io/npm/dm/nnls.svg
[download-url]: https://www.npmjs.com/package/nnls
