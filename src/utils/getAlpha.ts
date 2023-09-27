import { Matrix } from 'ml-matrix';

/**
 * When we get here, we are sure that some z_j is less than 0
 * so setting alpha in first line is fine.
 */
export function getAlpha({ x, z, P }: { x: Matrix; z: Matrix; P: Uint8Array }) {
  let alpha = x.get(0, 0) / (x.get(0, 0) - z.get(0, 0));
  for (let i = 0; i < P.length; i++) {
    if (P[i] && z.get(i, 0) <= 0) {
      const ratio = x.get(i, 0) / (x.get(i, 0) - z.get(i, 0));
      if (ratio < alpha) alpha = ratio;
    }
  }
  return alpha;
}
