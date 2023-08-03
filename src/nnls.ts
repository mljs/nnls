import { Matrix } from 'ml-matrix';

import { optimize } from './optimize';
import { checkInputDimensions, maxWiFromZ } from './utils';

interface Options {
  /**
   * Maximum number of iterations.
   * @default 3 * nCoefficients
   */
  maxIterations?: number;
}

/**
 * Find `x` that minimizes the distance ||Ax - b|| with x >= 0.
 * 1. Starts \vec{x} = 0,
 * 2. Compute the gradients,
 * 3. Solve for the positive gradients.
 * Nomenclature: "l" is a vector, "L" is a matrix.
 * "Lt" denotes L transpose.
 * @param X - input data
 * @param y - response data
 * @param options - options
 */

export function nnls(
  X: number[][] | Matrix,
  y: number[] | Matrix,
  options: Options = {},
) {
  const { E, f } = checkInputDimensions(X, y); // E=data, f=response.

  const { columns: nCoefficients /*rows: nEquations*/ } = E;
  let { maxIterations = 20 * nCoefficients } = options;

  // step 1
  let x = Matrix.zeros(nCoefficients, 1); // unknowns
  const Z = new Uint8Array(nCoefficients).fill(1); // 1s
  const P = new Uint8Array(nCoefficients); //0s

  // precompute part of g = Et(f - E * x)
  const Et = E.transpose();
  const EtE = Et.mmul(E); //square matrix
  const Etf = Et.mmul(f); //column vector like f.

  while (maxIterations) {
    // step 2
    const w = Etf.sub(EtE.mmul(x)); // g

    // step 3, 4
    const { indexOfMaxW, stop } = maxWiFromZ(Z, w);
    if (stop) {
      break;
    } else {
      // step 5
      P[indexOfMaxW] = 1;
      Z[indexOfMaxW] = 0;
    }

    x = optimize({
      E,
      f,
      Z,
      P,
      x,
      w,
      indexOfMaxW,
    });
    maxIterations--;
  }
  const dual = Etf.sub(EtE.mmul(x));
  return {
    resultVector: x.to1DArray(),
    dualVector: dual.to1DArray(),
    residualVector: E.mmul(x).sub(f).to1DArray(),
  };
}
