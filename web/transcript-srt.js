import {uiText} from "./i18n.js";

/** Offer a separate simplified copy without replacing the native transcript or export link. */
export function mountTranscriptSrt(find, request) {
  const button = find("media-srt");
  const script = find("media-srt-script");
  const status = find("media-srt-status");
  let current;
  let pending = false;

  function showStatus(message) {
    status.textContent = message;
    status.hidden = !message;
  }

  script.onchange = () => showStatus("");
  button.onclick = async event => {
    if (current?.language !== "zh" || script.value !== "simplified") return;
    event.preventDefault();
    if (pending) return;
    const report = current;
    pending = true;
    showStatus(uiText("Готовим упрощённые субтитры…"));
    try {
      const result = await request("subtitles/simplify", {language: report.language, srt: report.srt});
      if (report !== current || script.value !== "simplified") return;
      const url = URL.createObjectURL(new Blob([result.srt], {type: "application/x-subrip"}));
      const download = document.createElement("a");
      download.href = url;
      download.download = "subtitles.zh-Hans.srt";
      download.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      showStatus("");
    } catch (error) {
      if (report === current && script.value === "simplified") showStatus(error.message);
    } finally {
      pending = false;
    }
  };

  return function update(report) {
    current = report;
    if (button.hasAttribute("href")) URL.revokeObjectURL(button.href);
    script.hidden = report?.language !== "zh";
    if (report) {
      button.href = URL.createObjectURL(new Blob([report.srt], {type: "application/x-subrip"}));
      button.hidden = false;
    } else {
      button.removeAttribute("href");
      button.hidden = true;
      script.value = "original";
    }
    showStatus("");
  };
}
