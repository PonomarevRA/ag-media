import {uiText} from "./i18n.js";
import {packMedia} from "./person-layer.js";
import {accentStyleLabel, voiceProfileLabel} from "./voice-catalog.js";

/** Enroll short, reviewed acoustic references and listen to the saved accent takes. */
export function mountAccentProfiles(find, request) {
  const panel = find("voice-accent-profiles");
  const form = find("voice-accent-form");
  const style = find("voice-accent-style");
  const language = find("voice-accent-language");
  const name = find("voice-accent-name");
  const file = find("voice-accent-file");
  const text = find("voice-accent-text");
  const save = find("voice-accent-save");
  const status = find("voice-accent-status");
  const list = find("voice-accent-list");
  const audio = find("voice-accent-preview");
  let loading = 0;

  function chooseLanguage(reset) {
    const native = style.value === "german" ? "de" : "ja";
    for (const option of language.options) {
      option.hidden = option.disabled = ![native, "ru", "en"].includes(option.value);
    }
    if (reset) language.value = native;
  }

  async function refresh() {
    const generation = ++loading;
    try {
      const {profiles} = await request("voice-catalog", undefined, {purpose: "dubbing"});
      if (generation !== loading) return;
      list.replaceChildren();
      for (const profile of profiles.filter(row => row.accentStyle)) {
        const row = document.createElement("li");
        row.className = "candidate";
        const label = document.createElement("span");
        label.textContent = `${voiceProfileLabel(profile)} · ${accentStyleLabel(profile.accentStyle)}`;
        const preview = document.createElement("button");
        preview.type = "button";
        preview.textContent = uiText("Прослушать");
        preview.onclick = () => {
          audio.src = `/api/voice-accent-preview?version=${profile.version}`;
          audio.hidden = false;
          audio.play().catch(error => { status.textContent = error.message; });
        };
        row.append(label, preview);
        list.append(row);
      }
    } catch (error) {
      if (generation === loading) status.textContent = error.message;
    }
  }

  chooseLanguage(false);
  style.onchange = () => chooseLanguage(true);
  panel.ontoggle = () => {
    if (panel.open) void refresh();
    else audio.pause();
  };
  form.onsubmit = async event => {
    event.preventDefault();
    const source = file.files[0];
    if (!source || save.disabled) return;
    save.disabled = true;
    status.textContent = uiText("Проверяем аудиообразец…");
    try {
      await request("voice-accent-profile", packMedia(source, {
        style: style.value, referenceLanguage: language.value, name: name.value.trim(), text: text.value.trim(),
      }));
      form.reset();
      chooseLanguage(true);
      status.textContent = uiText("Профиль сохранён. Выбери его для спикера в дубляже.");
      await refresh();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      save.disabled = false;
    }
  };
}
