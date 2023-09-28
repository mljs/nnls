# nnls

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Non-Negative Least-Squares (NNLS) algorithm, by Lawson and Hanson. It was mostly done for learning purposes, and will be improved over time.

You are welcome to report issues and contribute to this project.

Find details in the [Misc.](#Misc.) section below.

## Installation

`$ npm i nnls`

## Usage

```js
import { nnls } from 'nnls';

const { resultVector/*, dualVector, residualVector*/ } = nnls(X, y);
// use `resultVector.to1DArray()` and so on... 
//to get the result as a 1D array
```

Like other implementations (for example `scipy.optimize.nnls`) it is limited to a single vector $y$, or as it is called in the literature, a single right hand side.

For multiple right hand sides, you can take a look at [Fast-Combinatorial Non-Negative Least-Squares](https://github.com/mljs/fcnnls)


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
