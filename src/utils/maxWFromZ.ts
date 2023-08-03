import { Matrix } from 'ml-matrix';

/**
 * There are two tasks here:
 * 1. Determine if Z is empty (all zeros)
 * 2. Get the max gradient in w
 * Max gradient from Z
 * If Z is empty
 * If all gradients from Z indices <=0, it stops
 */
export function maxWiFromZ(Z: Uint8Array, w: Matrix) {
  let indexOfMaxW = 0;
  let setZIsEmpty = true;
  for (let i = 0; i < w.rows; i++) {
    if (Z[i]) {
      setZIsEmpty = false;
      if (w.get(i, 0) > w.get(indexOfMaxW, 0)) {
        indexOfMaxW = i;
      }
    }
  }
  const negativeMaxGradient = w.get(indexOfMaxW, 0) <= 0;
  return {
    indexOfMaxW,
    stop: setZIsEmpty || negativeMaxGradient,
  };
}
