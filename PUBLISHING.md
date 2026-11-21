# Preparing private and public repositories

Use the public profile for a public source repository. The private profile adds
the architecture and audio processing guides for an internal repository.

Run from the project root with Python 3.12 or later:

```sh
python3 tools/export_repository.py --profile public --check-only
python3 tools/export_repository.py --profile private --check-only
python3 tools/export_repository.py --profile public --destination "$TMPDIR/ag-media-public-source"
python3 tools/export_repository.py --profile private --destination "$TMPDIR/ag-media-private-source"
```

Use fresh destinations outside the source directory. A public source archive
contains only the inputs for its public profile; the private profile additionally
requires the named internal source documents. Each result contains a
`repository/` tree, `source.zip`, and `audit.json`. Extract the ZIP into an empty
folder for a candidate repository. The archive tree is the input for a new
repository's first commit.

`packaging/export-profiles.json` is the reviewed allowlist. Both profiles exclude
credentials, environment files, user media, transcripts, model weights, installed
dependencies, runtime state, caches, screenshots, reports and existing Git
history. Both profiles include the getting-started and agent guides, bundled agent memory and
workflow skills and portable synthetic tests. Private research tests stay
outside both profiles. Review the allowlist before adding new files. The
exporter rejects symlinks and forbidden paths in selected source.

Selected files are read as UTF-8 source and scanned before writing. Findings
contain relative paths and rule IDs, without matching values. The rules cover
common credential formats, private keys, credential URLs, account home paths,
e-mail addresses and the active nongeneric host account name. This is a bounded automated
check, not proof that arbitrary personal information or all secrets are absent.
Review the actual selected source and dependency rights before any remote upload.

For releases prepared from a workspace that held private media, create a UTF-8
review file outside the repository. List one literal filename, identifier or
distinctive example label per line. Pass its path to
`--reject-references-file` on the check and export commands. The exporter
matches source text and archive paths without regard to case or Unicode
composition. Findings omit the matching value. Keep the review file local.

The manifest records SHA-256 for every selected source file. Its own SHA-256 and
the ZIP SHA-256 are in `audit.json`. The ZIP has sorted entries, fixed timestamps,
and normalized permissions. Repeating preparation with the same selected bytes,
profile, Python and zlib versions yields identical bytes. Compression libraries
can differ between environments; the file manifest is the portable comparison.

Run the exporter contracts independently:

```sh
python3 -B -m unittest discover -s tests -p test_repository_export.py
```

A public source archive can be checked or rebuilt with the public profile.
Preparing a private source archive requires the full source tree or a private
archive containing the extra documentation.

For application installation and native building follow `README.md` and
`docs/NATIVE-DISTRIBUTION.md`. No model files, native engine binaries, CPython or
third-party packages are included in these source archives. Training datasets and
historical reports are deliberately absent; training and corpus evaluations need
separate authorized fixtures. A source archive is distinct from a signed native
installer and from a license-cleared redistributable binary release.

Original project source is licensed under MIT in `LICENSE`, as selected by the
project owner. Keep upstream licenses, copyright notices and model-specific terms
intact; MIT for project code does not relicense third-party code or resources. Review
`THIRD_PARTY_NOTICES.md`, the preserved `THIRD-PARTY.md`, and
`packaging/resource-license-inventory.json` when assembling a binary release.
