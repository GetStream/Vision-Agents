import SwiftUI
import VisionAgentsCore
import VisionAgentsUI

/// A written conversation.
///
/// Nothing is transcribed and nothing is spoken, so no call is joined. Everything between
/// hearing a question and answering it is the same as on a call: the same instructions, the
/// same knowledge, the same skills and the same two tools running on this phone.
///
/// `ConversationView` draws the approval card `refund_order` asks for, and the toolbar opens
/// the same conversation as AG-UI protocol events.
struct ChatView: View {
    let agent: String

    @State private var session: AgentSession?
    @State private var failure: String?
    @State private var isShowingEvents = false

    var body: some View {
        Group {
            if let session {
                ConversationView(session: session)
                    .toolbar {
                        Button("Events", systemImage: "list.bullet.rectangle") {
                            isShowingEvents = true
                        }
                    }
                    .sheet(isPresented: $isShowingEvents) {
                        NavigationStack {
                            EventsView(session: session)
                        }
                    }
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
                session = try await Demo.agents.chat(agent: agent, tools: Demo.tools)
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
