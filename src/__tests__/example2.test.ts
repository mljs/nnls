import { it } from 'vitest';

import { nnls } from '../nnls';

import { assertResult } from './test-utils';

it('Example 2', () => {
  const X = [
    [1, 1, 2],
    [10, 11, -9],
    [-1, 0, 0],
    [-5, 6, -7],
  ];
  const y = [-1, 11, 0, 1];
  const solution = [0.461, 0.5611, 0];
  const result = nnls(X, y);
  assertResult(result, solution);
});
