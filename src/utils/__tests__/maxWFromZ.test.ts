import { Matrix } from 'ml-matrix';
import { expect, describe, it } from 'vitest';

import { maxWiFromZ } from '..';

describe('Test how the gradient is selected', () => {
  it('All zeros', () => {
    const Z = new Uint8Array(10);
    const w = Matrix.zeros(10, 1);
    const result = maxWiFromZ(Z, w);
    expect(result.indexOfMaxW).toBe(0);
    expect(result.maxW).toBe(0);
  });
  it('All Z-zeros test 2', () => {
    const Z = new Uint8Array(10);
    const w = Matrix.zeros(10, 1);
    w.set(5, 0, 1);
    const result = maxWiFromZ(Z, w);
    expect(result.indexOfMaxW).toBe(0);
    expect(result.maxW).toBe(0);
  });
  it('One non-zero', () => {
    const Z = new Uint8Array(10);
    Z[5] = 1;
    const w = Matrix.zeros(10, 1);
    w.set(5, 0, 1);
    const result = maxWiFromZ(Z, w);
    expect(result.indexOfMaxW).toBe(5);
    expect(result.maxW).toBe(1);
  });
  it('Many non-zero', () => {
    const Z = new Uint8Array(10).fill(1);
    const w = Matrix.columnVector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const result = maxWiFromZ(Z, w);
    expect(result.indexOfMaxW).toBe(9);
    expect(result.maxW).toBe(10);
  });
});
