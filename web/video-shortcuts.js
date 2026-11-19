/** Prefer declared source FPS; otherwise retain the existing analysis timeline's millisecond grid. */
export function videoFrameStepMs(video, fallback = 40) {
  const fps = Number(video.dataset?.sourceFrameRate);
  if (Number.isFinite(fps) && fps > 0 && fps <= 240) return 1000 / fps;
  const timeline = Number(video.dataset?.editorFrameStepMs);
  if (Number.isFinite(timeline) && timeline > 0) return timeline;
  return Number.isFinite(fallback) && fallback > 0 ? fallback : 40;
}

/** Return a grid position strictly before EOF so End never requests an empty EOF frame. */
export function lastVideoFrameTime(video, fallback = 40) {
  const step = videoFrameStepMs(video, fallback);
  return Math.max(0, Math.floor((video.duration * 1000 - .001) / step) * step / 1000);
}

/** Determine whether a native media transport action is currently meaningful. */
export function videoActionEnabled(video, action, frameStepMs = 40) {
  if (video.hidden || video.readyState < 1 || !video.videoWidth || !Number.isFinite(video.duration)) return false;
  if (["start", "previous"].includes(action)) return video.currentTime > 0;
  if (action === "end") return Math.abs(video.currentTime - lastVideoFrameTime(video, frameStepMs)) > .000001;
  if (action === "next") return video.currentTime < lastVideoFrameTime(video, frameStepMs) - .000001;
  return action === "play";
}

/** Apply the available time grid directly to the source clock without replacing its URL or report. */
export function performVideoAction(video, action, frameStepMs = 40, secondsStep = false) {
  if (!videoActionEnabled(video, action, frameStepMs)) return false;
  if (action === "play") {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
    return true;
  }
  video.pause();
  const step = secondsStep ? 1 : videoFrameStepMs(video, frameStepMs) / 1000;
  const target = {previous: video.currentTime - step, next: video.currentTime + step, start: 0, end: lastVideoFrameTime(video, frameStepMs)}[action];
  video.currentTime = Math.min(lastVideoFrameTime(video, frameStepMs), Math.max(0, target));
  return true;
}

/** Keep transport shortcuts away from forms, buttons, open dialogs and other studio panels. */
export function mountVideoShortcuts(video, panel, frameStep) {
  document.addEventListener("keydown", event => {
    if (event.defaultPrevented || event.isComposing || event.metaKey || event.ctrlKey || event.altKey
        || panel.hidden || video.hidden || video.readyState < 1 || !video.videoWidth || !Number.isFinite(video.duration)
        || document.querySelector("dialog[open]")) return;
    const target = event.target;
    if (target !== document.body && target !== document.documentElement && !panel.contains(target)) return;
    if (target.closest?.("input, textarea, select, button, a[href], summary, [contenteditable], [role=button], [role=separator]")) return;
    const action = {ArrowLeft: "previous", ArrowRight: "next", Home: "start", End: "end"}[event.key];
    if (event.code === "Space" && !event.shiftKey) {
      event.preventDefault();
      if (!event.repeat) performVideoAction(video, "play", frameStep());
    } else if (action) {
      event.preventDefault();
      performVideoAction(video, action, frameStep(), event.shiftKey);
    }
  });
}
