#!/usr/bin/env bash
# Writes a2ui/instructions.md and openui/instructions.md from their prompt.json, with the A2UI and
# OpenUI renderers of stream-chat-swift, so the agent is told exactly what the app renders:
#
#     prompts/generate.sh [stream-chat-swift checkout]
#
# The checkout defaults to one beside this repository. Only the formats' own code is built, for
# this Mac: the StreamChatAI renderers around it need UIKit.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
assistant="$(dirname "$here")"
sdk="$(cd "${1:-$assistant/../../../../stream-chat-swift}" && pwd)"
for module in StreamChatA2UI StreamChatOpenUI; do
  if [[ ! -d "$sdk/Sources/$module" ]]; then
    echo "$sdk has no Sources/$module: check out its add/openui-module branch, or a release with it." >&2
    exit 1
  fi
done

package="${TMPDIR:-/tmp}/swift-assistant-prompts"
mkdir -p "$package/Sources"
ln -sfn "$sdk/Sources/StreamChatA2UI" "$package/Sources/StreamChatA2UI"
ln -sfn "$sdk/Sources/StreamChatOpenUI" "$package/Sources/StreamChatOpenUI"
ln -sfn "$here/a2ui-prompt" "$package/Sources/a2ui-prompt"
ln -sfn "$here/openui-prompt" "$package/Sources/openui-prompt"
cat > "$package/Package.swift" <<'SWIFT'
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SwiftAssistantPrompts",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    targets: [
        .target(name: "StreamChatA2UI", exclude: ["AIUI", "Info.plist"]),
        .target(name: "StreamChatOpenUI", exclude: ["GenerativeUI", "Info.plist"], resources: [.process("Resources")]),
        .executableTarget(name: "a2ui-prompt", dependencies: ["StreamChatA2UI"]),
        .executableTarget(name: "openui-prompt", dependencies: ["StreamChatOpenUI"])
    ]
)
SWIFT

swift build --package-path "$package" -c release >&2
bin="$(swift build --package-path "$package" -c release --show-bin-path)"

# Each file is replaced only once its instructions are written, so a broken example keeps the old one.
for format in a2ui openui; do
  "$bin/$format-prompt" "$assistant/$format/prompt.json" > "$assistant/$format/instructions.md.new"
  mv "$assistant/$format/instructions.md.new" "$assistant/$format/instructions.md"
done
