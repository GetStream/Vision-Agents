#!/usr/bin/env bash
# Writes a2ui/instructions.md and openui/instructions.md from their prompt.json, with the A2UI and
# OpenUI renderers of stream-chat-swift, so the agent is told exactly what the app renders:
#
#     prompts/generate.sh [stream-chat-swift checkout]
#
# The checkout defaults to one beside this repository. Only the formats' own code is built, for
# this Mac: the StreamChatAI renderers around it need UIKit. Each generator is built into the same
# module as its format's code, as the prompt builders they use aren't public.
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
rm -rf "$package/Sources/StreamChatA2UI" "$package/Sources/a2ui-prompt" "$package/Sources/StreamChatOpenUI" "$package/Sources/openui-prompt"
for format in a2ui openui; do
  mkdir -p "$package/Sources/$format-prompt"
  ln -sfn "$here/$format-prompt/main.swift" "$package/Sources/$format-prompt/main.swift"
done
ln -sfn "$sdk/Sources/StreamChatA2UI" "$package/Sources/a2ui-prompt/StreamChatA2UI"
ln -sfn "$sdk/Sources/StreamChatOpenUI" "$package/Sources/openui-prompt/StreamChatOpenUI"
cat > "$package/Package.swift" <<'SWIFT'
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SwiftAssistantPrompts",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "a2ui-prompt", exclude: ["StreamChatA2UI/AIUI", "StreamChatA2UI/Info.plist"]),
        .executableTarget(
            name: "openui-prompt",
            exclude: ["StreamChatOpenUI/GenerativeUI", "StreamChatOpenUI/Info.plist"],
            resources: [.process("StreamChatOpenUI/Resources")]
        )
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
