import type { Matrix } from 'ml-matrix';
/**
 * Important step (11) where we move the 1s from P to Z if x_i = 0
 * It's the only mechanism that removes from P.
 * @param - Indices Arrays (P,Z) and Coefficient vector x {@link UpdateIndices}
 */
export function updateIndices({ Z, P, x }: UpdateIndices) {
  for (let i = 0; i < P.length; i++) {
    // if gradient is positive and x=0 move it to Z.
    if (P[i] && x.get(i, 0) <= 0) {
      // see paper further discussion, it's `<=0` not `===0`
      Z[i] = 1;
      P[i] = 0;
    }
  }
}
export interface UpdateIndices {
  Z: Uint8Array;
  P: Uint8Array;
  x: Matrix;
}
