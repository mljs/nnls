import { type Matrix, solve } from 'ml-matrix';

import { shouldWeOptimize, getAlpha, updateIndices, makeEp } from './utils';
/**
 * Step 6
 * 1. Selects cols of E that are in P
 * 2. Writes them to Ep
 * 3. Solve Ep*z = f ( as standard least squares problem. )
 * Where columns of E selected by P indices are written in Ep. (data-like.)
 * @param - Solver object.
 * @returns - the solution z.
 */
export function optimize({ E, f, Z, P, x, w, indexOfMaxW }: Solver) {
  let positive = P.filter((p) => p === 1).length;
  let firstRun = true;
  while (positive) {
    const Ep = makeEp(E, P); // E(P) => Ep; P is a column mask.
    /*
     * Solve the subproblem E_p x_p = f, with x_z set to zero.
     */
    const z = solve(Ep, f, true);

    // max gradient is positive, check.
    if (firstRun) {
      if (z.get(indexOfMaxW, 0) <= 0) {
        // similar to fcnnls
        w.set(indexOfMaxW, 0, 0);
        return x;
      }
    }
    firstRun = false;

    if (shouldWeOptimize(Z, P, z)) return z; // back to step 2

    // if prev step was false
    const alpha = getAlpha(x, z, P);
    x = x.add(z.sub(x).mul(alpha));

    updateIndices({
      Z,
      P,
      x,
    });
    positive--;
  }
  return x;
}

export interface Solver {
  /**
   * data
   */
  E: Matrix;
  /**
   * response
   */
  f: Matrix;
  /**
   * working set of coefficients that are zero
   */
  Z: Uint8Array;
  /**
   * working set of coefficients that are positive
   */
  P: Uint8Array;
  /**
   * coefficients approximation
   */
  x: Matrix;
  /** Gradient */
  w: Matrix;
  indexOfMaxW: number;
}
