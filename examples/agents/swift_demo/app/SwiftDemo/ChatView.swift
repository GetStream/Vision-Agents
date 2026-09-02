import SwiftUI
import VisionAgentsCore
import VisionAgentsUI

/// A written conversation.
///
/// Nothing is transcribed and nothing is spoken, so no call is joined. Everything between
/// hearing a question and answering it is the same as on a call: the same instructions, the
/// same knowledge, the same skills and the same `lookup_order` running on this phone.
struct ChatView: View {
    let agent: String

    @State private var session: AgentSession?
    @State private var failure: String?

    var body: some View {
        Group {
            if let session {
                ConversationView(session: session, prompt: "Ask about an order")
            } else if let failure {
                ContentUnavailableView(
                    "Could not start", systemImage: "exclamationmark.triangle", description: Text(failure))
            } else {
                ProgressView()
            }
        }
        .task {
            guard session == nil else { return }
            do {
                session = try await Demo.agents.chat(agent: agent, tools: [Demo.lookupOrder])
            } catch is CancellationError {
                return
            } catch {
                failure = error.localizedDescription
            }
        }
        .onDisappear {
            let closing = session
            session = nil
            Task { await closing?.close() }
        }
    }
}
