import { Matrix } from 'ml-matrix';

import { solver } from './solver';
import { getRootSquaredError, checkInputDimensions, maxWiFromZ } from './utils';

export interface NnlsOptions<T extends boolean | undefined> {
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
  /**
   * Return information of the error at each iteration step
   * from the main iteration loop.
   * @default false
   */
  info?: T;
}

/**
 * Find $x$ that minimizes the distance `||Ax - b|| s.t x >= 0`
 * @param X - input data
 * @param y - response data
 * @param options - options
 */

export function nnls(
  X: number[][] | Matrix,
  y: number[] | Matrix,
  options?: NnlsOptions<false | undefined>,
): DataOnly;
export function nnls(
  X: number[][] | Matrix,
  y: number[] | Matrix,
  options?: NnlsOptions<true>,
): DataAndInfo;
export function nnls<T extends boolean | undefined>(
  X: number[][] | Matrix,
  y: number[] | Matrix,
  options?: NnlsOptions<T>,
): DataAndInfo | DataOnly;
export function nnls<T extends boolean | undefined>(
  X: number[][] | Matrix,
  y: number[] | Matrix,
  options: NnlsOptions<T> = {},
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

  // error before iterating.
  const error: number[] = [getRootSquaredError(E, f, x)];
  while (maxIterations--) {
    // step 2 - compute w
    const EtfClone = Etf.clone();
    const w = EtfClone.sub(EtE.mmul(x)); // g

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
    if (options.info) {
      error.push(getRootSquaredError(E, f, x));
    }
  }
  if (maxIterations === 0) {
    throw new Error('Maximum number of iterations reached.');
  }
  const dual = Etf.sub(EtE.mmul(x));
  if (options.info) {
    return {
      resultVector: x,
      dualVector: dual,
      info: {
        rse: error,
        nIterations: error.length,
      },
    };
  }
  return {
    resultVector: x,
    dualVector: dual,
  };
}

export interface Info {
  /**
   * Root Squared Error.
   * This is a row vector, the RSE values for each column of Y.
   */
  rse: number[];
  /**
   * The number of times K was calculated.
   */
  nIterations: number;
}
export type NNLS = DataAndInfo | DataOnly;
export interface DataAndInfo {
  resultVector: Matrix;
  dualVector: Matrix;
  info: Info;
}
export interface DataOnly {
  resultVector: Matrix;
  dualVector: Matrix;
}
