import { Matrix } from 'ml-matrix';

import { solver } from './solver';
import { checkInputDimensions, maxWiFromZ } from './utils';

interface NnlsOptions {
  /**
   * Maximum number of iterations.
   * @default 3 * nCoefficients
   */
  maxIterations?: number;
  /**
   * If false, you will be fitting `f(x)=C*X+b` instead of `f(x)=C*X`.
   * If you did this step manually, just leave it as true.
   * @default true
   */
  interceptAtZero?: boolean;
}

/**
 * Find `x` that minimizes the distance `||Ax - b||` s.t `x >= 0`.
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
  options: NnlsOptions = {},
) {
  const goodInput = checkInputDimensions(X, y); // E=data, f=response.
  let E = goodInput.E;
  const f = goodInput.f;

  // Add intercept
  if (options.interceptAtZero === false) {
    E = Matrix.ones(E.rows, E.columns + 1).setSubMatrix(E, 0, 1);
  }
  const { columns: nCoefficients /*rows: nEquations*/ } = E;
  let { maxIterations = 3 * nCoefficients } = options;

  // step 1
  let x = Matrix.zeros(nCoefficients, 1); // unknowns
  const Z = new Uint8Array(nCoefficients).fill(1); // 1s
  const P = new Uint8Array(nCoefficients); //0s

  // pre-compute part of g = Et(f - E.x)
  const Et = E.transpose();
  const EtE = Et.mmul(E); //square matrix
  const Etf = Et.mmul(f); //column vector like f.

  while (maxIterations--) {
    // step 2 - compute w
    const w = Etf.sub(EtE.mmul(x)); // g
    // step 3A
    if (!Z.some((z) => z !== 0)) {
      break;
    }
    // Step 3B and 4
    const { indexOfMaxW, maxW } = maxWiFromZ(Z, w);
    if (maxW <= 0) {
      break;
    }

    // step 5 - We are sure that maxW > 0
    P[indexOfMaxW] = 1;
    Z[indexOfMaxW] = 0;

    x = solver({
      E,
      f,
      Z,
      P,
      x,
      w,
      indexOfMaxW,
    });
  }
  const dual = Etf.sub(EtE.mmul(x));
  return {
    resultVector: x.to1DArray(),
    dualVector: dual.to1DArray(),
    residualVector: E.mmul(x).sub(f).to1DArray(),
  };
}
