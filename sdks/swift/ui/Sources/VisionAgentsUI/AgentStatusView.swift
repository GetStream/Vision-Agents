import SwiftUI
import VisionAgentsCore

/// What the agent is doing, in a line.
public struct AgentStatusView: View {
    private let state: Conversation.State

    public init(state: Conversation.State) {
        self.state = state
    }

    public var body: some View {
        HStack(spacing: 6) {
            if busy {
                ProgressView().controlSize(.mini)
            }
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .animation(.default, value: label)
    }

    private var busy: Bool {
        switch state {
        case .responding, .working, .listening: return true
        case .idle, .ended: return false
        }
    }

    private var label: String {
        switch state {
        case .idle: return "ready"
        case .listening: return "listening"
        case .responding: return "answering"
        case .working(let skills):
            return skills.isEmpty
                ? "thinking" : "thinking (\(skills.joined(separator: ", ")))"
        case .ended: return "the conversation ended"
        }
    }
}
