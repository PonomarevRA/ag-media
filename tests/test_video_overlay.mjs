import test from "node:test";
import assert from "node:assert/strict";
import {frameAnnotation, paintTechnicalFrame, personCountAt, previewExportPlan, sceneFrameAt} from "../web/video-overlay.js";

test("person counts retain measured zero and do not cross a shot cut", () => {
  const samples = [{timeMs: 0, count: 2}, {timeMs: 1120, count: 0}];
  assert.equal(personCountAt(samples, [0, 800], 150), 2);
  assert.equal(personCountAt(samples, [0, 800], 795), null);
  assert.equal(personCountAt(samples, [0, 800], 900), 0);
  assert.equal(personCountAt(samples, [], 800, [0, 1000]), null);
  assert.equal(personCountAt(samples, [0, 800], 2000), null);
});

test("speech moves to subtitles when the camera cuts away or a second face is active", () => {
  const speaker = [1, .1, .1, .3, .3, 1];
  const listener = [2, .5, .1, .3, .3, 0];
  const phrase = {startMs: 0, endMs: 120, text: "Ну, это правда.", speaker: 1};
  const evidence = {
    timeline: {stepMs: 40, frames: [[speaker], [listener], [speaker, [...listener.slice(0, 5), 1]]]},
    phrases: {0: {activity: {state: "visible-active", track: 1}}},
    faceSpeakers: {1: {speaker: 1}},
  };
  assert.equal(frameAnnotation(evidence, [phrase], 20).speakerFace, 1);
  for (const time of [40, 80]) {
    const frame = frameAnnotation(evidence, [phrase], time);
    assert.equal(frame.speakerFace, undefined);
    assert.equal(frame.phrase, phrase);
  }
  assert.equal(frameAnnotation(evidence, [phrase], phrase.endMs).phrase, undefined);
});

test("both faces highlight only inside a sustained simultaneous interval", () => {
  const faces = [[1, .1, .1, .3, .3, 1], [2, .5, .1, .3, .3, 1]];
  const evidence = {
    timeline: {stepMs: 40, frames: Array(8).fill(faces)},
    simultaneousFaces: [{startMs: 40, endMs: 280, tracks: [1, 2]}],
    faceSpeakers: {1: {speaker: 1}, 2: {speaker: 2}},
  };
  const cues = [{startMs: 0, endMs: 320, overlap: true,
    overlaps: [{startMs: 40, endMs: 280, speakers: [1, 2]}]}];
  assert.deepEqual(frameAnnotation(evidence, cues, 20).speakingFaces, []);
  assert.deepEqual(frameAnnotation(evidence, cues, 40).speakingFaces, [1, 2]);
  assert.deepEqual(frameAnnotation(evidence, cues, 279).speakingFaces, [1, 2]);
  assert.deepEqual(frameAnnotation(evidence, cues, 280).speakingFaces, []);
});

test("a sustained face highlights inside a longer unconfirmed cue while a one-frame response does not", () => {
  const talking = [2, .05, .05, .3, .3, 1, 9];
  const frames = [[], [talking], [], [talking], [talking], [talking], [talking], [talking], [], []];
  const phrase = {startMs: 0, endMs: 400, text: "тест", speakerNumber: 3};
  const evidence = {timeline: {stepMs: 40, frames}, phrases: {}, faceSpeakers: {9: {speaker: 3}}};
  assert.equal(frameAnnotation(evidence, [phrase], 40).speakerFace, undefined);
  assert.equal(frameAnnotation(evidence, [phrase], 200).speakerFace, 2);
  assert.equal(frameAnnotation(evidence, [{...phrase, speakerNumber: 4}], 200).speakerFace, undefined);
});

const measure = {measureText: text => ({width: text.length * 6})};

test("technical labels follow the exact depth interval and track identity", () => {
  const evidence = {timeline: {stepMs: 40, frames: [[[2, .1, .2, .3, .4, 0, 6]]]},
    editorTimeline: {personMotion: [
      {track: 2, faceId: 8, startMs: 0, endMs: 40, depth: "foreground"},
      {track: 2, faceId: 6, startMs: 10, endMs: 30, depth: "background", depthBasis: "relative-face-scale"},
    ]}};
  const {samples} = previewExportPlan(measure, evidence, [], [], [], 480, 360, 40);
  assert.deepEqual(samples.map(([start, end, commands]) =>
    [start, end, commands.find(item => item[0] === "text")[5]]), [
    [0, 10, "Лицо 6"], [10, 30, "Лицо 6 · Задний план ≈"], [30, 40, "Лицо 6"],
  ]);
});

test("export resolves imported overlapping captions by their original order", () => {
  const cues = [
    {startMs: 120, endMs: 340, text: "Первый", speaker: 1},
    {startMs: 30, endMs: 290, text: "Второй", speaker: 2},
    {startMs: 360, endMs: 430, text: "Третий", speaker: 3},
  ];
  const {samples} = previewExportPlan(measure, {simultaneousFaces: null}, cues, [], [], 480, 360, 500);
  assert.deepEqual(samples.map(([start, end, commands]) => [
    start, end, commands.find(command => command[0] === "text")[5],
  ]), [
    [30, 120, "Спикер 2 · Второй"],
    [120, 340, "Спикер 1 · Первый"],
    [360, 430, "Спикер 3 · Третий"],
  ]);
});

test("export highlights only the first matching simultaneous face interval", () => {
  const faces = [[1, .1, .1, .3, .3, 1], [2, .5, .1, .3, .3, 1]];
  const evidence = {
    timeline: {stepMs: 40, frames: Array(8).fill(faces)},
    simultaneousFaces: [{startMs: 80, endMs: 160, tracks: [2]}, {startMs: 40, endMs: 240, tracks: [1]}],
    faceSpeakers: {1: {speaker: 1}, 2: {speaker: 2}},
  };
  const cues = [{startMs: 0, endMs: 320, overlap: true, text: "Да.",
    overlaps: [{startMs: 0, endMs: 320, speakers: [1, 2]}]}];
  const {samples} = previewExportPlan(measure, evidence, cues, [], [], 400, 320, 320);
  const highlights = samples.map(([start, end, commands]) => {
    const outline = commands.find(command => command[0] === "stroke" && command[1] === "#ffb34e");
    return [start, end, outline?.[2]];
  });
  assert.deepEqual(highlights, [[0, 40, undefined], [40, 80, 40], [80, 160, 200],
    [160, 240, 40], [240, 320, undefined]]);
});

test("MP4 reuses the preview labels and splits an active outline at the speaker boundary", () => {
  const face = [1, .1, .1, .4, .4, 1, 5];
  const evidence = {timeline: {stepMs: 40, frames: [[face]]}, faceSpeakers: {5: {speaker: 2}},
    phrases: {0: {activity: {state: "visible-active", track: 1}}},
    personLayer: {frames: [{timeMs: 0, people: [{track: 1, headPose: {horizontal: "right", vertical: "up"},
      gaze: {horizontal: "left"}}]}]}};
  const cues = [{startMs: 0, endMs: 20, speakerNumber: 2, text: "Да."},
    {startMs: 20, endMs: 40, speakerNumber: 3, text: "Нет."}];
  const objects = [{timeMs: 0, objects: [{box: [.7, .5, .1, .2]}]}];
  const {samples} = previewExportPlan(measure, evidence, cues, objects, [], 480, 360, 40);
  assert.deepEqual(samples.map(row => row.slice(0, 2)), [[0, 20], [20, 40]]);
  assert(samples[0][2].some(row => row[0] === "text" && row[5] === "Спикер 2 · Взгляд ← · Голова →↑"));
  assert(samples[0][2].some(row => row[0] === "text" && row[5] === "Спикер 2 · Да."));
  assert(samples[0][2].some(row => row[0] === "stroke" && row[1] === "rgba(153, 225, 152, .50)"));
  assert(samples[0][2].some(row => row[0] === "stroke" && row[1] === "#ffb34e"));
  assert(!samples[1][2].some(row => row[1] === "#ffb34e"));
});

test("compact previews keep the face label clear of the caption plate", () => {
  const face = [1, .15, .16, .26, .36, 1, 1];
  const evidence = {timeline: {stepMs: 40, frames: [[face]]}, faceSpeakers: {1: {speaker: 1}},
    phrases: {0: {activity: {state: "visible-active", track: 1}}},
    personLayer: {frames: [{timeMs: 0, people: [{track: 1, gaze: {horizontal: "left"},
      headPose: {horizontal: "right", vertical: "up"}}]}]}};
  const cues = [{startMs: 0, endMs: 40, speaker: 1, text: "Тестовая реплика."}];
  const {samples} = previewExportPlan(measure, evidence, cues, [], [], 290, 164, 40);
  const plates = samples[0][2].filter(row => row[0] === "fill" && row[1].startsWith("rgba(16"));
  assert.equal(plates.length, 2);
  assert(plates[0][3] + plates[0][5] < plates[1][3], JSON.stringify(plates));
});

test("quiet object marks follow nearby measured frames without crossing cuts", () => {
  const present = {timeMs: 100, objects: [{box: [.2, .2, .1, .1]}]};
  const empty = {timeMs: 280, objects: []};
  assert.equal(sceneFrameAt([present, empty], [150], 145), present);
  assert.equal(sceneFrameAt([present, empty], [150], 151), null);
  assert.equal(sceneFrameAt([present, empty], [150], 200), empty);
  assert.equal(sceneFrameAt([present, empty], [150], 400), null);
});

test("export includes an object in a late shot independently of the face timeline", () => {
  const evidence = {timeline: {stepMs: 40, frames: [[]]}};
  const frames = [{timeMs: 62100, objects: [{box: [.2, .2, .1, .1]}]}];
  const {samples} = previewExportPlan(measure, evidence, [], frames, [62050], 480, 360, 65000);
  assert.deepEqual(samples.map(row => row.slice(0, 2)), [[62050, 62200]]);
  assert(samples[0][2].some(row => row[0] === "stroke" && row[1] === "rgba(153, 225, 152, .50)"));
});

test("export includes short-lived OCR outlines and stops them at the next cut", () => {
  const text = [{timeMs: 120, text: [{box: [.1, .2, .16, .08]}]}];
  const {samples} = previewExportPlan(measure, null, [], [], [180], 480, 360, 400, text);
  assert.deepEqual(samples.map(([start, end]) => [start, end]), [[21, 180]]);
  const stroke = samples[0][2].find(row => row[0] === "stroke" && row[1] === "rgba(199, 173, 251, .48)");
  assert.deepEqual(stroke?.at(-1), [5, 3]);
});

test("a body rectangle and gaze are never carried into a different shot track", async () => {
  const {personMarks} = await import("../web/video-overlay.js");
  const frame = {timeMs: 0, people: [{track: 1, box: [.2, .2, .5, .7], gaze: {horizontal: "left"}}]};
  const visible = [[1, .3, .3, .2, .2, 1, 5]];
  const otherShot = [[9, .3, .3, .2, .2, 1, 5]];
  assert.deepEqual(personMarks({frames: [frame]}, visible, 100).get(1).box, [.2, .2, .5, .7]);
  assert.equal(personMarks({frames: [frame]}, otherShot, 100).size, 0);
  assert.equal(personMarks({frames: [frame]}, visible, 210).get(1).box, undefined);
});

test("monitor details report the marks painted in the current shot or selected sample", () => {
  const ctx = new Proxy(measure, {get: (target, method) => target[method] ?? (() => {})});
  const box = [0.1, 0.2, 0.3, 0.4];
  const sample = {timeMs: 90, objects: Array(4).fill({label: "cup", box}),
    text: Array(5).fill({text: "Studio", box})};
  const cuts = [0, 100];
  const details = (time, selected) => {
    let frame;
    paintTechnicalFrame(ctx, null, [], [sample], cuts, time, 480, 360, selected, [sample], undefined,
      current => { frame = current; });
    return frame;
  };
  const first = details(90);
  assert.equal(first.shot, 1);
  assert.equal(first.objects.length, 3);
  assert.equal(first.text.length, 4);
  const nextShot = details(100);
  assert.equal(nextShot.shot, 2);
  assert.equal(nextShot.objects, undefined);
  assert.equal(nextShot.text, undefined);
  const inspected = details(90, sample);
  assert.equal(inspected.objects.length, 4);
  assert.equal(inspected.text.length, 5);
});

test("a verified interface outranks embedded marks and respects a hard cut in the export", () => {
  const panel = {box: [.63, 0, .37, 1], panelType: "mobile-ui"};
  const inside = {label: "small", box: [.7, .3, .06, .1]};
  const objects = [{timeMs: 1250, objects: [inside,
    {label: "small", box: [.1, .3, .01, .01]},
    {label: "prominent", box: [.1, .4, .3, .32]}]}];
  const panels = [{timeMs: 1000, interfaces: [panel]}];
  const text = [{timeMs: 1250, text: [inside]}];
  const ctx = new Proxy(measure, {get: (target, method) => target[method] ?? (() => {})});
  let details;
  paintTechnicalFrame(ctx, null, [], objects, [1500], 1250, 480, 360, null, text, undefined,
    current => { details = current; }, panels);
  assert.equal(details.panels.length, 1);
  assert.deepEqual(details.objects.map(row => row.label), ["prominent", "small"]);
  assert.deepEqual(details.text, []);
  const {samples} = previewExportPlan(measure, null, [], objects, [1500], 480, 360, 2000, text, panels);
  const outlined = samples.filter(([, , commands]) => commands.some(command =>
    command[0] === "stroke" && command[1] === "rgba(152, 191, 255, .85)"));
  assert.equal(outlined[0][0], 251);
  assert.equal(outlined.at(-1)[1], 1500);
});

test("inspecting a measured frame keeps the phone and discards the TV inside the verified panel", () => {
  const panel = {box: [.63, 0, .37, 1], panelType: "mobile-ui"};
  const phone = {label: "cell phone", box: [.21, .55, .12, .30]};
  const tv = {label: "tv", box: [.65, .03, .31, .92]};
  const sample = {timeMs: 1000, panelChecked: true, interfaces: [panel], objects: [tv, phone], text: []};
  const ctx = new Proxy(measure, {get: (target, method) => target[method] ?? (() => {})});
  let details;
  paintTechnicalFrame(ctx, null, [], [sample], [], 1000, 960, 540, sample, [], undefined,
    frame => { details = frame; }, [sample]);
  assert.deepEqual(details.objects, [phone]);
  assert.deepEqual(details.panels, [panel]);
});

test("measured object tracks hold through sparse frames, update on a measurement and stop at a cut", () => {
  const box = [.6, .55, .1, .25];
  const moved = [.65, .55, .1, .25];
  const isolated = [.1, .1, .1, .1];
  const clips = [
    {label: "tie", startMs: 100, endMs: 426, boxSamples: [[100, box], [300, moved], [425, moved]]},
    {label: "cup", startMs: 200, endMs: 201, boxSamples: [[200, isolated]]},
  ];
  const frames = [
    {timeMs: 100, objects: [{label: "tie", box}]},
    {timeMs: 200, objects: [{label: "cup", box: isolated}]},
    {timeMs: 300, objects: [{label: "tie", box: moved}]},
    {timeMs: 425, objects: [{label: "tie", box: moved}]},
  ];
  const {samples} = previewExportPlan(measure, null, [], frames, [400], 100, 100, 500, [], [], clips);
  const positionsAt = time => samples.find(([start, end]) => start <= time && time < end)?.[2].map(row => row[2]) ?? [];
  assert.deepEqual(positionsAt(99), []);
  assert.deepEqual(positionsAt(100), [60]);
  assert.deepEqual(positionsAt(200), [60, 10]);
  assert.deepEqual(positionsAt(299), [60]);
  assert.deepEqual(positionsAt(300), [65]);
  assert.deepEqual(positionsAt(399), [65]);
  assert.deepEqual(positionsAt(400), []);
  assert.deepEqual(positionsAt(425), [65]);
  assert.deepEqual(positionsAt(426), []);
});

test("change boxes are exported only during the measured source frame", () => {
  const witness = {startMs: 680, endMs: 720, regions: [{box: [.2, .3, .04, .05], changedPixels: 100}]};
  const args = [measure, {}, [], [], [], 400, 200, 1600, [], [], undefined];
  assert.deepEqual(previewExportPlan(...args).samples, []);
  const {samples} = previewExportPlan(...args, [witness]);
  assert.deepEqual(samples, [[680, 720, [["stroke", "#ffdf66", 80, 60, 16, 10, 1.5, [3, 3]]]]]);
});
