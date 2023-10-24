import { Matrix, QrDecomposition, SingularValueDecomposition } from 'ml-matrix';

import {
  shouldWeOptimize,
  getAlpha,
  updateIndices,
  maskToIndices,
} from './utils';
/**
 * Step 6
 * This step does not follow the paper that much to improve speed.
 * @param Solver object.
 * @returns The solution.
 */
export function solver({ Z, P, x, w, indexOfMaxW, E, f }: Solver) {
  // Note that for P=zeros the algorithm skips loop
  const colsToSolve = P.filter((p) => p === 1).length;
  const z: Matrix = Matrix.zeros(E.columns, 1);
  let reducedZ: Matrix;
  for (let i = 0; i < colsToSolve; i++) {
    const indices = maskToIndices(P);

    const EtoSolve = E.subMatrixColumn(indices);

    const solveQR = new QrDecomposition(EtoSolve);
    reducedZ = solveQR.isFullRank()
      ? (reducedZ = solveQR.solve(f))
      : new SingularValueDecomposition(EtoSolve).solve(f);

    // 6B Define z_i=0 for i in Z
    let pIndex = 0;
    for (let i = 0; i < Z.length; i++) {
      if (Z[i]) {
        z.set(i, 0, 0);
      } else {
        z.set(i, 0, reducedZ.get(pIndex, 0));
        pIndex++;
      }
    }

    // step 7
    const swop = shouldWeOptimize({ P, z });
    if (swop) return z; // back to step 2

    // test whenever it comes from step 5 (first cycle)
    if (i === 0 && z.get(indexOfMaxW, 0) <= 0) {
      w.set(indexOfMaxW, 0, 0);
      return x;
    }

    // if prev step was false
    const alpha = getAlpha({ x, z, P });
    x.add(z.sub(x).mul(alpha));

    updateIndices({
      Z,
      P,
      x,
    });
  }
  return x;
}

export interface Solver {
  /** working set of coefficients that are zero */
  Z: Uint8Array;
  /** working set of coefficients that are positive */
  P: Uint8Array;
  /** coefficients approximation */
  x: Matrix;
  /** Gradient */
  w: Matrix;
  indexOfMaxW: number;
  E: Matrix;
  f: Matrix;
}
