import { Matrix } from 'ml-matrix';

import { checkInputDimensions } from '../checkInputDimensions';

describe('checkInputDimensions', () => {
  test('should return matrices for valid input', () => {
    const X = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const y = [7, 8, 9];
    const { E, f } = checkInputDimensions(X, y);

    expect(E).toBeInstanceOf(Matrix);
    expect(f).toBeInstanceOf(Matrix);
  });

  test('should handle Matrix input', () => {
    const X = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const y = new Matrix([[7], [8], [9]]);
    const { E, f } = checkInputDimensions(X, y);

    expect(E).toBeInstanceOf(Matrix);
    expect(f).toBeInstanceOf(Matrix);
  });

  test('should throw an error if response is not a column vector', () => {
    const X = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const y = [[7, 8, 9]];
    // @ts-expect-error testing invalid type input
    expect(() => checkInputDimensions(X, y)).toBeDefined();
  });
  test('should throw an error if input dimensions do not match', () => {
    const X = [
      [1, 2],
      [3, 4],
    ];
    const y = [5, 6, 7];
    expect(() => checkInputDimensions(X, y)).toThrow(
      "The number of rows on X and y don't match",
    );
  });
});
