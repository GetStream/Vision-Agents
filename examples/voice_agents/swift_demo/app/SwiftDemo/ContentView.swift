import SwiftUI
import VisionAgentsCore

/// Talks to the agent, in writing or out loud.
///
/// There is no agent picker. Reading the configs is server-side only, so a device cannot ask
/// which agents exist; it is told which one it talks to, the way a real app would be, in
/// `Demo.agentName`.
struct ContentView: View {
    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Larkspur support")
                .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var content: some View {
        TabView {
            ChatView(agent: Demo.agentName)
                .tabItem { Label("Chat", systemImage: "text.bubble") }
            VoiceView(agent: Demo.agentName)
                .tabItem { Label("Voice", systemImage: "waveform") }
        }
    }
}
