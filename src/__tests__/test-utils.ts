import { Matrix } from 'ml-matrix';
import { expect } from 'vitest';

import { DataAndInfo, NNLSOutput } from '../nnls';

export function assertResult(
  output: NNLSOutput,
  solution: number[] | Matrix,
  precision = 4,
) {
  const result = output.x;
  solution = Array.isArray(solution) ? Matrix.columnVector(solution) : solution;
  for (let i = 0; i < result.rows; i++) {
    expect(result.get(i, 0)).toBeCloseTo(solution.get(i, 0), precision);
  }
}
export function testLastErrorValue(
  output: DataAndInfo,
  expected: number,
  precision = 4,
) {
  expect(output.info.rse[output.info.rse.length - 1]).toBeCloseTo(
    expected,
    precision,
  );
}
