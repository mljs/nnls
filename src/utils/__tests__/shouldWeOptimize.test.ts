import { Matrix } from 'ml-matrix';

import { shouldWeOptimize } from '../shouldWeOptimize';

describe('shouldWeOptimize', () => {
  it('should return true if all z_i > 0 for all i in P', () => {
    const P = new Uint8Array([1, 0, 1]);
    const z = Matrix.columnVector([1, 2, 3]);
    const result = shouldWeOptimize({ P, z });
    expect(result).toBe(true);
  });

  it('should return false if there exists i in P such that z_i <= 0', () => {
    const P = new Uint8Array([1, 0, 1]);
    const z = Matrix.columnVector([1, -2, 3]);
    const result = shouldWeOptimize({ P, z });
    expect(result).toBe(true);
  });

  it('should set z_i to 0 for all i in Z', () => {
    const P = new Uint8Array([1, 0, 1]);
    const z = Matrix.columnVector([-1, 2, 3]);
    expect(shouldWeOptimize({ P, z })).toBe(false);
  });
});
