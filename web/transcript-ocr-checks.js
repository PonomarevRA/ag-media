import {uiText} from "./i18n.js";

/** Keep OCR suggestions on the report clock and let the user open the evidence frame. */
export function mountTranscriptOcrChecks(find, video) {
  const root = find("media-ocr-checks");
  const list = find("media-ocr-check-list");
  let current;

  return function update(report) {
    const checks = report?.ocrTranscriptChecks;
    if (checks === current) return;
    current = checks;
    list.replaceChildren();
    for (const {status, timeMs, transcript, observed} of checks ?? []) {
      if (status !== "review") continue;
      const item = document.createElement("li");
      const seek = document.createElement("button");
      seek.type = "button";
      seek.className = "transcript-time";
      const seconds = timeMs / 1000;
      const centiseconds = Math.round(timeMs / 10);
      const minutes = Math.floor(centiseconds / 6000);
      const clock = `${minutes}:${(centiseconds % 6000 / 100).toFixed(2).padStart(5, "0")}`;
      seek.textContent = clock;
      seek.setAttribute("aria-label", uiText`Открыть кадр ${clock}`);
      seek.onclick = () => {
        if (video.readyState) video.currentTime = seconds;
      };
      item.append(seek, ` ${transcript} → ${observed}`);
      list.append(item);
    }
    root.hidden = !list.childElementCount;
  };
}
