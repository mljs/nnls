import { type Matrix, solve } from 'ml-matrix';

import { shouldWeOptimize, getAlpha, updateIndices, makeEp } from './utils';
/**
 * Step 6
 * 1. Selects cols of E that are in P
 * 2. Writes them to Ep
 * 3. Solve Ep*z = f ( as standard least squares problem. )
 * Where columns of E selected by P indices are written in Ep. (data-like.)
 * The remaining E(Z) will stay at 0.
 * So we solve a problem equivalent to setting some xs (xz) to 0.
 * And approximating the rest of xs (xp) with a least squares problem.
 * @param - Solver object.
 * @returns - The solution.
 */
export function optimize({ E, f, Z, P, x, w, indexOfMaxW }: Solver) {
  const colsToSolve = P.filter((p) => p === 1).length;

  for (let i = 0; i < colsToSolve; i++) {
    const Ep = makeEp(E, P); // E(P) => Ep; P is a column mask.

    // Solve E_p x = f, with E(Z) = 0.
    const z = solve(Ep, f, true);

    // max gradient is positive, check.
    if (i === 0 && z.get(indexOfMaxW, 0) <= 0) {
      // similar to fcnnls
      w.set(indexOfMaxW, 0, 0);
      return x;
    }

    if (shouldWeOptimize(Z, P, z)) return z; // back to step 2

    // if prev step was false
    const alpha = getAlpha(x, z, P);
    x = x.add(z.sub(x).mul(alpha));

    updateIndices({
      Z,
      P,
      x,
    });
  }
  return x;
}

export interface Solver {
  /** data */
  E: Matrix;
  /** response */
  f: Matrix;
  /** working set of coefficients that are zero */
  Z: Uint8Array;
  /** working set of coefficients that are positive */
  P: Uint8Array;
  /** coefficients approximation */
  x: Matrix;
  /** Gradient */
  w: Matrix;
  indexOfMaxW: number;
}
