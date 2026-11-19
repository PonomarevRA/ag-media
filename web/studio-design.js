import {mountStudioShell} from "./studio-shell.js";
const profileKey = "voice-studio:design-profile";
const colorKey = "voice-studio:color-mode";
const recipes = {
  retro: {player: "cassette", recorder: "cassette", deck: "hi-fi", timeline: "amber"},
  modern: {player: "compact", recorder: "meter", deck: "flat", timeline: "studio"},
  editor: {player: "transport", recorder: "monitor", deck: "console", timeline: "workbench"},
};
const colorModes = ["system", "light", "dark"];

const retroSignal = {ink: "#3e210c", peaks: "#623210", cursor: "#211109", grid: "#965111", processed: "#783614"};
const palettes = {
  retro: {light: retroSignal, dark: retroSignal},
  modern: {
    light: {ink: "#654689", peaks: "#795aa1", cursor: "#332547", grid: "#ad9dbe", processed: "#a04a50"},
    dark: {ink: "#c7b5fb", peaks: "#a89cce", cursor: "#eeeaf8", grid: "#534b67", processed: "#f5bda3"},
  },
  editor: {
    light: {ink: "#126454", peaks: "#287985", cursor: "#172e3a", grid: "#7897a2", processed: "#3b7136"},
    dark: {ink: "#6ee1d0", peaks: "#70b8ca", cursor: "#e8eef5", grid: "#576d83", processed: "#8dec9f"},
  },
};

/** Resolve high-contrast canvas colors against the current recorder or deck surface. */
export function studioSignalPalette(document) {
  const {designProfile, colorMode} = document.documentElement.dataset;
  return palettes[designProfile]?.[colorMode] ?? retroSignal;
}

/** Select the visual face for a recorder, player, deck or timeline node. */
function applySurface(node, recipe) {
  node.dataset.presentation = recipe[node.dataset.designComponent];
}

/** Apply a profile to mounted components without disturbing audio or input state. */
function setProfile(profile, document) {
  document.documentElement.dataset.designProfile = profile;
  for (const surface of document.querySelectorAll("[data-design-component]")) {
    applySurface(surface, recipes[profile]);
  }
  document.dispatchEvent(new Event("studio-design-change"));
}

/** Restore an allowed browser preference, including when storage is disabled. */
function readChoice(key, choices, fallback) {
  try {
    const value = localStorage.getItem(key);
    return choices.includes(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function saveChoice(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

/** Restore independent profile and color preferences and follow OS changes in system mode. */
export function installStudioDesign(document) {
  mountStudioShell(document);
  const profile = readChoice(profileKey, Object.keys(recipes), "retro");
  const appearance = document.getElementById("studio-color-mode");
  const prefersDark = matchMedia("(prefers-color-scheme: dark)");
  appearance.value = readChoice(colorKey, colorModes, "dark");

  /** Resolve the effective appearance before redrawing existing canvas surfaces. */
  function updateColors() {
    const choice = appearance.value;
    document.documentElement.dataset.colorMode = choice === "system" ? (prefersDark.matches ? "dark" : "light") : choice;
    document.dispatchEvent(new Event("studio-design-change"));
  }

  updateColors();
  setProfile(profile, document);
  appearance.addEventListener("change", () => {
    saveChoice(colorKey, appearance.value);
    updateColors();
  });
  prefersDark.addEventListener("change", () => {
    if (appearance.value === "system") updateColors();
  });
  for (const option of document.querySelectorAll('input[name="studio-design"]')) {
    option.checked = option.value === profile;
    option.addEventListener("change", () => {
      setProfile(option.value, document);
      saveChoice(profileKey, option.value);
    });
  }
}

/** Create the player face for the active profile and leave playback to mountPlayer. */
export function createPlayerSurface(template) {
  const player = template.content.firstElementChild.cloneNode(true);
  const profile = template.ownerDocument.documentElement.dataset.designProfile;
  applySurface(player, recipes[profile] ?? recipes.retro);
  return player;
}
