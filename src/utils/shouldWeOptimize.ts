import { Matrix } from 'ml-matrix';

/**
 * If column went to positive stop optimizing it.
 */
export function shouldWeOptimize({ P, z }: { P: Uint8Array; z: Matrix }) {
  for (let i = 0; i < P.length; i++) {
    if (P[i] && z.get(i, 0) <= 0) {
      //if some z is negative for P>0, stop optimizing.
      return false;
    }
  }
  return true;
}
