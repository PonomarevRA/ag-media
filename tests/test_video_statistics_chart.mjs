import test from "node:test";
import assert from "node:assert/strict";
import {statisticsBinAt, statisticsYTop, formatStatisticsValue, statisticsClock} from "../web/video-statistics-chart.js";
import {buildVideoStatistics} from "../web/video-statistics.js";

test("plot coordinates clamp to exact first/last bins and ignore invalid geometry", () => {
  assert.equal(statisticsBinAt(0, 40, 300, 10), 0);
  assert.equal(statisticsBinAt(40, 40, 300, 10), 0);
  assert.equal(statisticsBinAt(66, 40, 300, 10), 1);
  assert.equal(statisticsBinAt(299.9, 40, 300, 10), 9);
  assert.equal(statisticsBinAt(300, 40, 300, 10), 9);
  assert.equal(statisticsBinAt(400, 40, 300, 10), 9);
  for (const value of [NaN, Infinity]) assert.equal(statisticsBinAt(value, 40, 300, 10), null);
  assert.equal(statisticsBinAt(70, 40, 40, 10), null);
  assert.equal(statisticsBinAt(70, 40, 300, 0), null);
});

test("responsive widths retain the same source bin for the same normalized source position", () => {
  for (const width of [180, 320, 947]) {
    for (const count of [1, 29, 60, 120]) {
      for (const i of [0, Math.floor(count / 2), count - 1]) {
        const x = 30 + width * (i + .5) / count;
        assert.equal(statisticsBinAt(x, 30, 30 + width, count), i);
      }
    }
  }
});

test("duration axes fit sub-second measurements while count axes stay integral", () => {
  assert.equal(statisticsYTop(.5, true), .5);
  assert.equal(statisticsYTop(.21, true), .5);
  assert.equal(statisticsYTop(.02, true), .02);
  assert.equal(statisticsYTop(2.1, true), 5);
  assert.equal(statisticsYTop(.5, false), 1);
  assert.equal(statisticsYTop(3, false), 3);
  assert.equal(statisticsYTop(0, true), 1);
});

test("a real one-millisecond observation remains distinct from zero and missing data", () => {
  const model = buildVideoStatistics({durationMs: 1000, videoEvidence: {facialExpressions: {intervals: [
    {label: "happy", startMs: 100, endMs: 101, faceId: 1, track: 1, confidence: .9},
  ]}}});
  assert.equal(model.expressions.series[0].values[0], .001);
  assert.equal(formatStatisticsValue(.001, true), "1 мс");
  assert.equal(formatStatisticsValue(.0025, true), "2,5 мс");
  assert.equal(formatStatisticsValue(.004, true), "4 мс");
  assert.equal(formatStatisticsValue(0, true), "0 с");
  assert.equal(formatStatisticsValue(null, true), "Нет измерений");
  assert.equal(formatStatisticsValue(undefined, true), "Нет измерений");
  assert.equal(formatStatisticsValue(2), "2");
});

test("interval captions preserve milliseconds and correctly carry at minute boundaries", () => {
  assert.equal(statisticsClock(0), "0:00.000");
  assert.equal(statisticsClock(1), "0:00.001");
  assert.equal(statisticsClock(59999.6), "1:00.000");
  assert.equal(statisticsClock(7200123), "120:00.123");
});
