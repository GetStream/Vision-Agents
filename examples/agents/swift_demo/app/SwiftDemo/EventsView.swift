import AGUI
import SwiftUI
import VisionAgentsCore

/// The same conversation as AG-UI protocol events, newest first.
///
/// `session.aguiEvents()` is the whole of it: the SDK translates what the router publishes
/// into the protocol, so an app already written against AG-UI reads a Vision Agents session
/// without knowing whose router it is. Every caller gets a stream of its own, so opening this
/// takes nothing away from anybody else watching.
struct EventsView: View {
    let session: AgentSession

    @State private var lines: [Line] = []

    /// One event, in the two things worth showing about it.
    struct Line: Identifiable {
        let id = UUID()
        let type: String
        let detail: String
    }

    var body: some View {
        List(lines) { line in
            VStack(alignment: .leading, spacing: 2) {
                Text(line.type)
                    .font(.system(.caption, design: .monospaced))
                if !line.detail.isEmpty {
                    Text(line.detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .listStyle(.plain)
        .navigationTitle("AG-UI events")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if lines.isEmpty {
                ContentUnavailableView(
                    "Nothing yet", systemImage: "list.bullet.rectangle",
                    description: Text("Say something in the chat and the run turns up here."))
            }
        }
        .task {
            for await event in session.aguiEvents() {
                // Newest first, so there is nothing to scroll to. Capped, because a long
                // conversation is a long log and this is a window on it, not a record.
                lines.insert(Line(type: event.type.rawValue, detail: detail(of: event)), at: 0)
                lines = Array(lines.prefix(500))
            }
        }
    }

    private func detail(of event: AGUI.Event) -> String {
        switch event {
        case .runStarted(let started):
            return "run \(started.runId)"
        case .runFinished(let finished):
            let waiting = finished.interrupts.compactMap(\.message)
            return waiting.isEmpty ? "run \(finished.runId)" : waiting.joined(separator: ", ")
        case .runError(let failure):
            return failure.message
        case .textMessageStart(let start):
            return start.role.rawValue
        case .textMessageContent(let content):
            return content.delta
        case .toolCallStart(let call):
            return call.toolCallName
        case .toolCallArgs(let arguments):
            return arguments.delta
        case .toolCallResult(let result):
            return result.content
        case .activitySnapshot(let activity):
            return activity.content["skill"]?.stringValue ?? activity.activityType
        default:
            return ""
        }
    }
}
