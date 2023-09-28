import { type Matrix } from 'ml-matrix';

export function getRootSquaredError(E: Matrix, f: Matrix, x: Matrix) {
  const error = E.mmul(x).sub(f);
  let absoluteError = 0;
  for (let i = 0; i < error.rows; i++) {
    absoluteError += error.get(i, 0) ** 2;
  }
  return Math.sqrt(absoluteError);
}
