import {uiText} from "./i18n.js";

/** Include a stable version so profiles with the same name remain distinguishable. */
export function voiceProfileLabel(profile) {
  return profile.name ? `${profile.name} · v${profile.version}` : uiText`Профиль ${profile.version}`;
}

/** Show comparison evidence without interpreting cosine as a percentage. */
export function voiceProfileSummary(profile) {
  const parts = [uiText`Записей: ${profile.recordingCount}`];
  if (profile.metrics?.medianCosine != null) {
    const {medianCosine, minimumCosine, pairCount, activeSeconds} = profile.metrics;
    parts.push(uiText`Косинус медиана ${medianCosine.toFixed(3)}, минимум ${minimumCosine.toFixed(3)} · пар: ${pairCount}`);
    parts.push(uiText`Активная речь: ${activeSeconds.toFixed(1)} с`);
  } else {
    parts.push(uiText("Сходство образцов не измерено"));
  }
  return parts.join(" · ");
}

/** Keep profile controls synchronized with the selection used by synthesis. */
export function mountVoiceCatalog(find, selectProfile, playSample) {
  const list = find("voice-catalog-list");
  const name = find("voice-catalog-name");
  const picker = find("synthesis-voice-profile");
  let key;
  let namedVersion;
  let previousName;

  picker.onchange = () => selectProfile(Number(picker.value));
  return {
    render(profiles, version) {
      const active = profiles.find(row => row.version === version);
      if (version !== namedVersion || active?.name !== previousName) {
        name.value = active?.name || "";
        namedVersion = version;
        previousName = active?.name;
      }
      find("synthesis-voice-summary").textContent = active ? voiceProfileSummary(active) : "";
      const signature = JSON.stringify([profiles, version]);
      if (signature === key) return;
      key = signature;
      const focused = list.contains(document.activeElement) ? document.activeElement.dataset.version : null;
      list.replaceChildren();
      picker.replaceChildren(new Option(uiText("Текущий набор"), ""));
      for (const profile of profiles) {
        const row = document.createElement("li");
        row.className = "candidate voice-catalog-entry";
        row.dataset.agWidth = "fill";
        const use = document.createElement("button");
        use.type = "button";
        use.className = "voice-catalog-select";
        use.textContent = voiceProfileLabel(profile);
        use.dataset.version = profile.version;
        use.setAttribute("aria-pressed", String(version === profile.version));
        use.dataset.available = String(profile.synthesisReady);
        use.onclick = () => selectProfile(profile.version);
        const summary = document.createElement("p");
        summary.className = "muted";
        summary.textContent = voiceProfileSummary(profile);
        const actions = document.createElement("div");
        actions.className = "actions";
        actions.append(use);
        if (profile.previewId) {
          const listen = document.createElement("button");
          listen.type = "button";
          listen.textContent = uiText("Прослушать");
          listen.setAttribute("aria-label", uiText`Прослушать образец — ${voiceProfileLabel(profile)}`);
          listen.onclick = () => playSample(profile);
          actions.append(listen);
        }
        row.append(actions, summary);
        list.append(row);
        const option = new Option(voiceProfileLabel(profile), String(profile.version));
        option.disabled = !profile.synthesisReady;
        picker.add(option);
      }
      picker.value = active ? String(version) : "";
      find("voice-catalog-empty").hidden = Boolean(profiles.length);
      if (focused) list.querySelector(`[data-version="${focused}"]`)?.focus({preventScroll: true});
    },
    sync(profiles, version, working) {
      picker.disabled = working || !profiles.some(row => row.synthesisReady);
      name.disabled = working;
      find("voice-profile-rename").disabled = working || !version || !profiles.some(row => row.version === version);
      for (const button of list.querySelectorAll(".voice-catalog-select")) {
        button.disabled = working || button.dataset.available !== "true";
      }
    },
  };
}
