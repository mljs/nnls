import { Matrix } from 'ml-matrix';

import { nnls } from '../nnls';

import { data } from './sample_data/data1';
import { data2 } from './sample_data/data2';
import { data3 } from './sample_data/data3';
import { data4 } from './sample_data/data4';

function assertResult(result: number[], solution: number[], precision = 4) {
  for (let i = 0; i < result.length; i++) {
    expect(result[i]).toBeCloseTo(solution[i], precision);
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
    const y = [-1, -1, -1];
    const solution = [0, 0];
    const result = nnls(X, y);
    assertResult(result.resultVector, solution);
  });
  it('Example 6: dont intercept at zero', () => {
    const { X, Y, X0, Y0 } = data3;
    const result = nnls(X, Y, { interceptAtZero: false });
    const resultVector = result.resultVector;
    const result2 = nnls(X0, Y0);
    expect(resultVector[0]).toBe(0);
    resultVector.shift();
    assertResult(resultVector, result2.resultVector);
    // residual should be very small for almost exact results.
    assertResult(
      result.residualVector,
      [
        -8.881784197001252e-16, -8.881784197001252e-16, -1.7763568394002505e-15,
        -1.7763568394002505e-15, -1.7763568394002505e-15,
        -3.552713678800501e-15, -3.552713678800501e-15, -3.552713678800501e-15,
        -3.552713678800501e-15, -3.552713678800501e-15, -7.105427357601002e-15,
        -7.105427357601002e-15,
      ],
    );
  });
  it('Example 7: data2', () => {
    const { X, Y } = data2;
    const result = nnls(X, Y);
    assertResult(result.resultVector, [0, 0.00002958854986853723]);
  });
  it('Example 8: data4', () => {
    // can't be fully sure that this is the best solution.
    const { X, Y } = data4;
    const result = nnls(X, Y, { interceptAtZero: false });
    assertResult(
      result.resultVector,
      [4.870637450199203, 0, 0.7552191235059765],
    );
  });
});
