"""Estimate Russian sound coverage with eSpeak NG 1.52 via espeakng-loader."""
import ctypes
from pathlib import Path
import re

import espeakng_loader

from voice_studio.synthesis.corpus import EXTRA
from voice_studio.prosody.reading_story import STORY_PARTS, STORY_TEXT
from voice_studio.app.studio import PROMPTS


def phonemizer():
    """Open the local eSpeak library with explicit IPA phone separators."""
    library = ctypes.CDLL(espeakng_loader.get_library_path())
    data = str(Path(espeakng_loader.get_data_path()).parent).encode()
    library.espeak_Info.restype = ctypes.c_char_p
    if library.espeak_Initialize(2, 0, data, 0) < 0 or library.espeak_SetVoiceByName(b"ru") != 0:
        raise RuntimeError("Cannot initialize the Russian eSpeak NG voice.")
    library.espeak_TextToPhonemes.restype = ctypes.c_char_p
    library.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
    return library


def phone_units(library, text):
    """Count unique IPA symbols and adjacent phones within words per clause."""
    encoded = text.encode()
    cursor = ctypes.c_char_p(encoded)
    phones, diphones = set(), set()
    while cursor.value:
        before = cursor.value
        raw = library.espeak_TextToPhonemes(ctypes.byref(cursor), 1, 2 | (ord("|") << 8))
        for word in (raw or b"").decode().split():
            units = [unit for unit in re.sub(r"[ˈˌ]", "", word).split("|") if unit]
            phones.update(units)
            diphones.update(zip(units, units[1:]))
        if cursor.value == before:
            raise RuntimeError("eSpeak NG did not advance through the script.")
    return phones, diphones


def main():
    """Measure each passage and compare the same model with the current corpus."""
    library = phonemizer()
    print("eSpeak NG", library.espeak_Info(None).decode())
    baseline = " ".join(PROMPTS["ru"] + [text for _, text in EXTRA["ru"]])
    script_phone, script_pair = phone_units(library, STORY_TEXT)
    base_phone, base_pair = phone_units(library, baseline)
    for label, text in (*STORY_PARTS, ("История целиком", STORY_TEXT), ("Исходный корпус", baseline)):
        phone, pair = phone_units(library, text)
        print(f"{label:25} {len(text.split()):3} words {len(phone):2} phones {len(pair):3} diphones")
    print("New phones versus existing prompts:", " ".join(sorted(script_phone - base_phone)))
    print("New diphones versus existing prompts:", len(script_pair - base_pair))
    print("Not in story:", " ".join(sorted(base_phone - script_phone)))


if __name__ == "__main__":
    main()
