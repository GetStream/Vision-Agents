#!/usr/bin/env bash
# Writes basic_catalog.json and protocol.md, the A2UI basic catalog and protocol the Go SDK compiles
# `ui:` with when its format is a2ui, from the A2UI renderer of stream-chat-swift, so the agent is
# told what the apps render:
#
#     sdks/go/agents/a2ui/generate.sh [stream-chat-swift checkout]
#
# The checkout defaults to one beside this repository. Only StreamChatA2UI's own code is built, for
# this Mac, into the same module as main.swift, as the catalog's serializer and the prompt aren't
# public.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sdk="$(cd "${1:-$here/../../../../../stream-chat-swift}" && pwd)"
if [[ ! -d "$sdk/Sources/StreamChatA2UI" ]]; then
  echo "$sdk has no Sources/StreamChatA2UI: check out its develop branch, or a release with it." >&2
  exit 1
fi

package="${TMPDIR:-/tmp}/go-sdk-a2ui-catalog"
mkdir -p "$package/Sources/a2ui-catalog"
ln -sfn "$here/main.swift" "$package/Sources/a2ui-catalog/main.swift"
ln -sfn "$sdk/Sources/StreamChatA2UI" "$package/Sources/a2ui-catalog/StreamChatA2UI"
cat > "$package/Package.swift" <<'SWIFT'
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "GoSDKA2UICatalog",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "a2ui-catalog", exclude: ["StreamChatA2UI/AIUI", "StreamChatA2UI/Info.plist"])
    ]
)
SWIFT

swift build --package-path "$package" -c release >&2
bin="$(swift build --package-path "$package" -c release --show-bin-path)"
# Each file is replaced only once it is written, so a failed run keeps the old one.
"$bin/a2ui-catalog" > "$here/basic_catalog.json.new"
"$bin/a2ui-catalog" protocol > "$here/protocol.md.new"
mv "$here/basic_catalog.json.new" "$here/basic_catalog.json"
mv "$here/protocol.md.new" "$here/protocol.md"
echo "from stream-chat-swift $(git -C "$sdk" rev-parse --short HEAD)" >&2
