import { Matrix } from 'ml-matrix';

import { nnls } from '../nnls';

import { data } from './sample_data/data1';

function assertResult(result: number[], solution: number[]) {
  for (let i = 0; i < result.length; i++) {
    expect(result[i]).toBeCloseTo(solution[i], 4);
  }
}
describe('NNLS tests', () => {
  it('Example1', () => {
    const result = nnls(data.mC, data.bf).resultVector;
    const solution = [0, 50, 10];
    assertResult(result, solution);
  });
  it('Example 2', () => {
    const X = [
      [1, 1, 2],
      [10, 11, -9],
      [-1, 0, 0],
      [-5, 6, -7],
    ];
    const y = [-1, 11, 0, 1];
    const solution = [0.461, 0.5611, 0];
    const result = nnls(X, y).resultVector;
    assertResult(result, solution);
  });
  it('Example 3', () => {
    const X = Matrix.eye(3).mul(-1);
    const y = [1, 2, 3];
    const solution = [0, 0, 0];
    const result = nnls(X, y).resultVector;
    assertResult(result, solution);
  });
  it('Example 4', () => {
    const X = [
      [1, 0],
      [1, 0],
      [0, 1],
    ];
    const y = [2, 1, 1];
    const solution = [1.5, 1];
    const result = nnls(X, y).resultVector;
    assertResult(result, solution);
  });
  it('Example 5', () => {
    const X = [
      [1, 0],
      [1, 0],
      [0, 1],
    ];
    const y = [2, 1, 1];
    const solution = [0, 0];
    const result = nnls(X, y).resultVector;
    assertResult(result, solution);
  });
});
