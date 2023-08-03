import { Matrix } from 'ml-matrix';

import { maxWiFromZ } from '..';

describe('Test how the gradient is selected', () => {
  it('All positive', () => {
    const gradients = Matrix.ones(10, 1);
    const Z = new Uint8Array(gradients.rows).fill(1);
    const { stop, indexOfMaxW } = maxWiFromZ(Z, gradients);
    expect(stop).toEqual(false);
    expect(indexOfMaxW).toEqual(0);
  });
  it('Z is all zeros', () => {
    const gradients = Matrix.ones(10, 1);
    const Z = new Uint8Array(gradients.rows);
    const { stop, indexOfMaxW } = maxWiFromZ(Z, gradients);
    expect(stop).toEqual(true);
    expect(indexOfMaxW).toEqual(0);
  });
  it('all wi negative (or 0)', () => {
    const gradients = Matrix.columnVector([
      -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    ]);
    const Z = new Uint8Array(gradients.rows).fill(1);
    const { stop, indexOfMaxW } = maxWiFromZ(Z, gradients);
    expect(stop).toEqual(true);
    expect(indexOfMaxW).toEqual(gradients.rows - 1);
  });
});
