export function maskToIndices(mask: Uint8Array) {
  const indices: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      indices.push(i);
    }
  }
  return indices;
}
