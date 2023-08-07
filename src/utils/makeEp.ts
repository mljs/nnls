import { Matrix } from 'ml-matrix';

/**
 * @param E - the data matrix
 * @param P - index of optimizable columns
 * @returns E(P), selection of positive columns from E
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
