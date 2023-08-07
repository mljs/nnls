import { Matrix } from 'ml-matrix';

import { makeEp } from '../makeEp';

describe('makeEp', () => {
  it('should return a matrix with only the columns specified by P', () => {
    const E = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const P = new Uint8Array([1, 0, 1]);
    const expected = new Matrix([
      [1, 0, 3],
      [4, 0, 6],
      [7, 0, 9],
    ]);
    const result = makeEp(E, P);
    expect(result.to2DArray()).toEqual(expected.to2DArray());
  });

  it('should return a matrix with all zeros if P is all zeros', () => {
    const E = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const P = new Uint8Array([0, 0, 0]);
    const expected = Matrix.zeros(3, 3);

    const result = makeEp(E, P);
    expect(result.to2DArray()).toEqual(expected.to2DArray());
  });

  it('should return a matrix with all zeros if E is empty', () => {
    const E = new Matrix([]);
    const P = new Uint8Array([]);
    const expected = new Matrix([]);
    const result = makeEp(E, P);
    expect(result.to2DArray()).toEqual(expected.to2DArray());
  });
});
