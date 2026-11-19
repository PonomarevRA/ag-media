#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
privacy_root="${AG_PRIVACY_ROOT:-$HOME/Documents/ai/ag-privacy}"
core="${AG_PRIVACY_CORE:-$privacy_root/src/AgPrivacy.Cli/bin/Release/net10.0/AgPrivacy.Core.dll}"
test -f "$core" || { printf '%s\n' 'Set AG_PRIVACY_CORE to a built AgPrivacy.Core.dll.' >&2; exit 2; }
dotnet build adapters/person-mentions/PersonMentions.csproj --configuration Release \
  "-p:AgPrivacyCorePath=$core" --artifacts-path "$(python3 voice_paths.py dependencies)/person-mentions"
