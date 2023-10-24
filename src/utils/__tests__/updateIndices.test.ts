import { Matrix } from 'ml-matrix';
import { expect, describe, it } from 'vitest';

import { updateIndices } from '..';

describe('UpdateIndices', () => {
  it('No updates', () => {
    const Z = new Uint8Array(10).fill(1);
    const P = new Uint8Array(10);
    const x = Matrix.columnVector([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
    updateIndices({ P, Z, x });
    // here we expect that the indices are the same (because P is 0)
    expect(P).toEqual(new Uint8Array(10));
    expect(Z).toEqual(new Uint8Array(10).fill(1));
  });
  it('Two updated elements on each array', () => {
    const Z = new Uint8Array(10);
    const P = new Uint8Array(10).fill(1);
    P[8] = 15;
    const x = Matrix.columnVector([1, 2, 3, 4, 5, 6, 7, 8, 0, 0]);
    updateIndices({ P, Z, x });
    // here we expect that the indices are the same (because P is 0)
    expect(P).toEqual(new Uint8Array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]));
    expect(Z).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]));
  });
});
