import { Matrix } from 'ml-matrix';

/**
 * If column went to positive stop optimizing it.
 */
export function shouldWeOptimize(Z: Uint8Array, P: Uint8Array, z: Matrix) {
  for (let i = 0; i < Z.length; i++) {
    // for i of Z, set z_i=0
    if (Z[i]) z.set(i, 0, 0);
    // for i of P, if all z_i > 0, we will set x = z
    if (P[i] && z.get(i, 0) <= 0) return false;
  }
  return true;
}
