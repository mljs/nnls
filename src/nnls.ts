import { solve, Matrix } from "ml-matrix";

import {
  maxWFromZ,
  setEpColumns,
  shouldWeOptimize,
  assertSameLength,
  getAlpha,
} from "./utils";
import type { UserInput, Solver } from "./types";

export function nnls(userInput: UserInput) {
  const { array2D, array1D } = userInput;

  assertSameLength(array2D, array1D);

  const data = new Matrix(array2D);
  const response = Matrix.columnVector(array1D);

  const nOfVariables = data.columns;
  let maxIterations = 3 * nOfVariables;

  let coefficients = Matrix.rowVector(new Array(nOfVariables));
  let Z = new Uint8Array(nOfVariables).fill(1);
  let P = new Uint8Array(nOfVariables);

  const Ep = new Matrix(data.rows, nOfVariables);
  while (maxIterations--) {
    const guessF = data.mmul(coefficients); // no mutation
    const residual = Matrix.sub(response, guessF); // no mutation

    let w = data.transpose().mmul(residual); // working set of coefficients

    const { indexOfMaxW, allZeroes } = maxWFromZ(Z, w, nOfVariables);

    const done = allZeroes || w.get(indexOfMaxW, 0) <= 0;
    if (done) return coefficients;
    else {
      P[indexOfMaxW] = 1;
      Z[indexOfMaxW] = 0;
    }

    optimize({
      E: data,
      Ep,
      f: response,
      nOfVariables,
      Z,
      P,
      x: coefficients,
    });
  }
  return coefficients;
}

function optimize({ E, Ep, f, nOfVariables, Z, P, x }: Solver) {
  let maxIterations = 2 * nOfVariables;
  while (maxIterations--) {
    setEpColumns(E, Ep, P, nOfVariables);

    const z = solve(Ep, f); // approximates x

    if (shouldWeOptimize(Z, P, nOfVariables, z)) {
      x = z;
      return;
    }

    const alpha = getAlpha(x, z, P, nOfVariables);
    x = x.add(z.sub(x).mul(alpha));

    updateIndexes({
      Z,
      P,
      x,
      nOfVariables,
    });
  }
}

type UpdateIndexes = {
  Z: Uint8Array;
  P: Uint8Array;
  x: Matrix;
  nOfVariables: number;
};
function updateIndexes({ Z, P, x, nOfVariables }: UpdateIndexes) {
  for (let i = 0; i < nOfVariables; i++) {
    if (P[i] && x.get(i, 0) == 0) {
      Z[i] = 1;
      P[i] = 0;
    }
  }
}
