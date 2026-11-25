/** Index overlapping intervals and return their values in display order on a half-open timeline. */
export function timelineIntervalLookup(ranges) {
  const sorted = ranges.map(([start, end, item], order) => ({start: Number(start), end: Number(end), item, order}))
    .filter(({start, end}) => start < end)
    .sort((left, right) => left.start - right.start);
  const size = 2 ** Math.ceil(Math.log2(Math.max(1, sorted.length)));
  const maxEnd = new Float64Array(size * 2).fill(-Infinity);
  sorted.forEach(({end}, index) => { maxEnd[size + index] = end; });
  for (let index = size - 1; index > 0; index--) {
    maxEnd[index] = Math.max(maxEnd[2 * index], maxEnd[2 * index + 1]);
  }
  return time => {
    const matches = [];
    function visit(index, start, length) {
      if (maxEnd[index] <= time || sorted[start].start > time) return;
      if (length === 1) {
        matches.push(sorted[start]);
        return;
      }
      const half = length / 2;
      visit(index * 2, start, half);
      visit(index * 2 + 1, start + half, half);
    }
    if (sorted.length && Number.isFinite(time)) visit(1, 0, size);
    return matches.sort((left, right) => left.order - right.order).map(({item}) => item);
  };
}
