import { Matrix } from 'ml-matrix';

/**
 * The optimizable columns from E (stored in P) are set to the columns of Ep
 * @param E - the data matrix
 * @param Ep - the matrix to be set
 * @param P - index of optimizable columns
 * @param nOfVariables
 */
export function makeEp(E: Matrix, P: Uint8Array) {
  const Ep = Matrix.zeros(E.rows, E.columns);
  for (let i = 0; i < P.length; i++) {
    if (P[i]) {
      Ep.setColumn(i, E.getColumn(i));
    }
  }
  return Ep;
}
