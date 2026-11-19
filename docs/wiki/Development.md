# Changes, checks and wiki maintenance

[Русский](Development-ru.md) · [ag-media-studio](../../README.md)

## Change workflow

- Find the functional owner through Atlas, then verify current source. A static map is not a runtime test.
- Edit canonical `voice_studio/` modules and preserve compatibility launchers and tool/schema IDs.
- Run affected contract checks. Model experiments need fixed sources, seeds and modes with separate quality/time/RSS results.
- Update responsibilities and journeys when behaviour changes.
- Build/check the wiki and obtain source-based independent review.

## Documentation build

```text
python3 scripts/build_wiki.py
python3 scripts/build_wiki.py --check
.venv/bin/python -m unittest tests.test_wiki -v
```

The renderer accepts typed paragraphs, headings, lists, tables, code, links and images. Raw HTML is escaped. Links are limited to known pages and HTTPS; SVG assets reject active content. HTTP serves only generated allowlisted files.

## Evidence limits

Build and link checks establish documentation consistency, not model accuracy or full application accessibility. Historical experiments and open AG gates do not become PASS through a rename. Refer to source-local reports for measurement limits.

## Design provenance

The versioned wiki structure follows ag-design `docs/wiki/`: start, run, architecture, agent usage and checks. This edition uses native HTML navigation and original schematic SVGs. No real frames or third-party UI screenshots are copied.

- [Start](Home.md)
- [Module index](Modules.md)
