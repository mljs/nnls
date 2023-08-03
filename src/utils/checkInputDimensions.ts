import { Matrix } from 'ml-matrix';

/**
 * Check if the input dimensions are valid.
 * @param X - input data
 * @param y
 * @returns
 */
export function checkInputDimensions(
  X: number[][] | Matrix,
  y: number[] | Matrix,
) {
  const E = Matrix.checkMatrix(X);
  const f = Array.isArray(y) ? Matrix.columnVector(y) : y;
  if (!f.isColumnVector) {
    throw new RangeError('"y" must be a vector or flat array.');
  }
  if (E.rows !== f.rows) {
    throw new RangeError("The number of rows on X and y don't match");
  }

  return { E, f };
}
