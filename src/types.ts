import type { Matrix } from "ml-matrix";

export type Solver = {
  /**
   * data
   */
  E: Matrix;
  /**
   * subset P of data.
   */
  Ep: Matrix;
  /**
   * response
   */
  f: Matrix;
  /**
   * working set of coefficients that are zero
   */
  Z: Uint8Array;
  /**
   * working set of coefficients that are positive
   */
  P: Uint8Array;
  /**
   * coefficients approximation
   */
  x: Matrix;
  nOfVariables: number;
};

export type Optimize = {
  /**
   * data
   */
  E: Matrix;
  /**
   * response
   */
  f: Matrix;
  /**
   * working set of coefficients that are zero
   */
  Z: Uint8Array;
  /**
   * working set of coefficients that are positive
   */
  P: Uint8Array;
  /**
   * coefficients approximation
   */
  x: Matrix;
  nOfVariables: number;
};

export type UserInput = {
  array1D: number[];
  array2D: number[][];
};
