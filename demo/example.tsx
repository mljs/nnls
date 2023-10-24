import React from 'react';
import { nnls } from '../src/index';
import { Plot, LineSeries, Axis, Legend, Heading } from 'react-plot';

import { data } from '../src/__tests__/sample_data/data1';

const { mC, bf } = data;

const result = nnls(mC, bf,{info: true}).info

const plot = result.rse.map((errorValue, iterationNumber) => ({
  x: iterationNumber,
  y: errorValue,
}));
export const Example = () => (
  <Plot
    width={1000}
    height={1000}
    margin={{ bottom: 50, left: 90, top: 50, right: 100 }}
  >
    <Heading
      title="RSE over iteration"
      subtitle="NNLS"
    />
    <LineSeries data={plot} xAxis="x" yAxis="y" label="nnls-error" />
    <Axis
      id="x"
      position="bottom"
      label="Iteration Number"
      // displayPrimaryGridLines
      // max={Math.max(...x) * 1.1}
    />
    <Axis
      id="y"
      position="left"
      label="RSE"
      // displayPrimaryGridLines
      // max={Math.max(...y) * 1.1}
    />
    <Legend position="right" />
  </Plot>
);
