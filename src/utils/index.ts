import { Matrix } from "ml-matrix";
/**
 * Searches for the max value of W in the subset of indexes given by Z.
 */
export function maxWFromZ(Z: Uint8Array, w: Matrix, nOfVariables: number) {
  let indexOfMaxW = 0;
  let currentW = w.get(indexOfMaxW, 0);
  let allZeroes = true;
  for (let i = 0; i < nOfVariables; i++) {
    // calculation finishes if all are negative
    currentW = w.get(i, 0);
    if (currentW > 0) {
      if (Z[i] && currentW >= currentW) {
        indexOfMaxW = i;
        allZeroes = false;
      }
    }
  }

  return {
    indexOfMaxW,
    allZeroes,
  };
}
export function setEpColumns(
  E: Matrix,
  Ep: Matrix,
  P: Uint8Array,
  nOfVariables: number
) {
  for (let i = 0; i < nOfVariables; i++) {
    if (P[i]) {
      Ep.setColumn(i, E.getColumn(i));
    }
  }
}

export function shouldWeOptimize(
  Z: Uint8Array,
  P: Uint8Array,
  nOfVariables: number,
  z: Matrix
) {
  let passes = true;
  for (let i = 0; i < nOfVariables; i++) {
    if (Z[i]) z.set(i, 0, 0);
    if (P[i] && z.get(i, 0) <= 0) passes = false;
  }
  return passes;
}
export function assertSameLength(array2D: number[][], array1D: number[]) {
  if (array2D.length !== array1D.length) {
    throw new Error("array2D and array1D must have the same length");
  }
}

export function getAlpha(
  x: Matrix,
  z: Matrix,
  P: Uint8Array,
  nOfVariables: number
) {
  let alpha = x.get(0, 0) / (x.get(0, 0) - z.get(0, 0));
  for (let i = 0; i < nOfVariables; i++) {
    if (P[i] && z.get(i, 0)) {
      const ratio = x.get(i, 0) / (x.get(i, 0) - z.get(i, 0));
      if (alpha > ratio) alpha = ratio;
    }
  }
  return alpha;
}
