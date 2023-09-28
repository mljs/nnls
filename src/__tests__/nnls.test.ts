import { Matrix } from 'ml-matrix';

import { nnls } from '../nnls';

import { data } from './sample_data/data1';
import { data2 } from './sample_data/data2';
import { data3 } from './sample_data/data3';

function assertResult(
  result: number[] | Matrix,
  solution: number[] | Matrix,
  precision = 4,
) {
  result = Array.isArray(result) ? Matrix.columnVector(result) : result;
  solution = Array.isArray(solution) ? Matrix.columnVector(solution) : solution;
  for (let i = 0; i < result.rows; i++) {
    expect(result.get(i, 0)).toBeCloseTo(solution.get(i, 0), precision);
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
    expect(result.MSE).toBeCloseTo(1.73205, 4);
  });
  it('Example 6: compare method with data', () => {
    // X5 is the same as X, but with a column of 1s added to the left.
    const { X, Y, XOnes } = data3;
    const extraParameter = nnls(X, Y, { interceptAtZero: false });
    const resultVector = extraParameter.resultVector; //less good result than numpy, in both cases.
    assertResult(resultVector, nnls(XOnes, Y).resultVector);
    // other software (like scipy) yields
    // const scipyResult1 = Matrix.columnVector([
    //   4.92128988, 0.34302285, 0.58189576,
    // ]);
    const scipyError1 = 0.5740809582615467;
    //ours is 0.5956313955391651
    expect(extraParameter.MSE).toBeGreaterThan(scipyError1);

    const forceToZero = nnls(X, Y);
    const scipySolution = [0, 0.93375969];
    const scipyError = 11.562826502844006;
    expect(forceToZero.MSE).toBeCloseTo(scipyError, 4);
    assertResult(forceToZero.resultVector, scipySolution);
  });
  it('Example 7: data2', () => {
    const { X, Y } = data2;
    const result = nnls(X, Y);
    assertResult(result.resultVector, [0, 0.00002958854986853723]);
  });
});
