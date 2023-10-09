import { Matrix } from 'ml-matrix';

import { nnls } from '../nnls';

import { data } from './sample_data/data1';
import { data2 } from './sample_data/data2';
import { data3 } from './sample_data/data3';
import { assertResult, testLastErrorValue } from './test-utils';

<<<<<<< Updated upstream
function assertResult(
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
function testLastErrorValue(
  output: DataAndInfo,
  expected: number,
  precision = 4,
) {
  expect(output.info.rse[output.info.rse.length - 1]).toBeCloseTo(
    expected,
    precision,
  );
}
=======
>>>>>>> Stashed changes
describe('NNLS tests', () => {
  it('Example1', () => {
    const result = nnls(data.mC, data.bf);
    const solution = [0, 50, 10];
    assertResult(result, solution);
  });
  it('Example 3', () => {
    const X = Matrix.eye(3).mul(-1);
    const y = [1, 2, 3];
    const solution = [0, 0, 0];
    const result = nnls(X, y);
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
    const result = nnls(X, y);
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
    const result = nnls(X, y, { info: true });
    assertResult(result, solution);
    const rse = result.info.rse[result.info.rse.length - 1];
    expect(rse).toBeCloseTo(1.73205, 4);
  });
  it('Example 6: compare method with data', () => {
    // X5 is the same as X, but with a column of 1s added to the left.
    const { X, Y, XOnes } = data3;
    const extraParameter = nnls(X, Y, { interceptAtZero: false, info: true });
    assertResult(extraParameter, nnls(XOnes, Y).x);
    const scipyResult1 = Matrix.columnVector([
      4.92128988, 0.34302285, 0.58189576,
    ]);
    assertResult(extraParameter, scipyResult1);
    const scipyError1 = 0.5740809582615467;
    //ours is 0.5956313955391651
    const forceToZero = nnls(X, Y, { info: true });
    const scipySolution = [0, 0.93375969];
    const scipyError = 11.562826502844006;

    testLastErrorValue(extraParameter, scipyError1, 8);
    testLastErrorValue(forceToZero, scipyError, 8);
    assertResult(forceToZero, scipySolution);
  });
  it('Example 7: data2', () => {
    const { X, Y } = data2;
    const result = nnls(X, Y);
    assertResult(result, [0, 0.00002958854986853723]);
  });
  it('simple case', () => {
    const X = new Matrix([
      [1, 0],
      [2, 0],
      [3, 0],
      [0, 1],
    ]);
    const Y = Matrix.columnVector([1, 2, 3, 4]);
    const solution = Matrix.columnVector([1, 4]);
    const result = nnls(X, Y, { info: true });
    assertResult(result, solution);
    testLastErrorValue(result, 0, 8);
  });

  it('does not converge because of limited max iterations', () => {
    const X = new Matrix([
      [1, 0],
      [2, 0],
      [3, 0],
      [0, 1],
    ]);
    const Y = Matrix.columnVector([1, 2, 3, 4]);
    expect(() => nnls(X, Y, { info: true, maxIterations: 0 })).toThrow();
  });
  it('simple case that requires a negative coefficient', () => {
    const X = new Matrix([
      [1, 0],
      [2, 0],
      [3, 0],
      [0, -1],
    ]);
    const y = Matrix.columnVector([1, 2, 3, 4]);
    const solution = Matrix.columnVector([1, 0]);
    const result = nnls(X, y, { info: true });
    assertResult(result, solution);
    testLastErrorValue(result, 4, 8);
  });
});
