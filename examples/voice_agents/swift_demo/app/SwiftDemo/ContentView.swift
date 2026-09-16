import SwiftUI
import VisionAgentsCore

/// Talks to the agent, in writing or out loud.
///
/// There is no agent picker. Reading the configs is server-side only, so a device cannot ask
/// which agents exist; it is told which one it talks to, the way a real app would be. The id
/// goes in `Demo.agentID`, and `go run ./configure` prints it.
struct ContentView: View {
    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Larkspur support")
                .navigationBarTitleDisplayMode(.inline)
        }
    }

    @ViewBuilder private var content: some View {
        if Demo.agentID.isEmpty {
            ContentUnavailableView {
                Label("No agent yet", systemImage: "person.crop.circle.badge.questionmark")
            } description: {
                Text(
                    "Run `go run ./configure` in examples/voice_agents/swift_demo, "
                        + "then put the config id it prints in Demo.agentID.")
            }
        } else {
            TabView {
                ChatView(agent: Demo.agentID)
                    .tabItem { Label("Chat", systemImage: "text.bubble") }
                VoiceView(agent: Demo.agentID)
                    .tabItem { Label("Voice", systemImage: "waveform") }
            }
        }
    }
}
