import { Matrix } from 'ml-matrix';

/**
 * If column went to positive stop optimizing it.
 */
export function shouldWeOptimize({
  Z,
  P,
  z,
}: {
  Z: Uint8Array;
  P: Uint8Array;
  z: Matrix;
}) {
  for (let i = 0; i < Z.length; i++) {
    if (Z[i]) {
      z.set(i, 0, 0);
    } else if (P[i] && z.get(i, 0) <= 0) {
      return false;
    }
  }
  return true;
}
