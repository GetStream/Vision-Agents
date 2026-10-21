import SwiftUI
import VisionAgentsCore

/// Talks to the agent, in writing or out loud.
///
/// There is no agent picker. Reading the configs is server-side only, so a device cannot ask
/// which agents exist; it is told which one it talks to, the way a real app would be, in
/// `Demo.agentName`.
struct ContentView: View {
    private enum Tab {
        case chat, voice
    }

    @State private var tab = Tab.chat
    @State private var voiceStyle = VoiceStyle.orb
    @State private var isPickingStyle = false

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Larkspur support")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    if tab == .voice && !Demo.streamAPIKey.isEmpty {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button {
                                isPickingStyle = true
                            } label: {
                                Image(systemName: "slider.horizontal.3")
                            }
                            .accessibilityLabel("Voice style")
                        }
                    }
                }
                .sheet(isPresented: $isPickingStyle) {
                    VoiceStyleSheet(style: $voiceStyle)
                        .presentationDetents([.height(320)])
                }
        }
    }

    @ViewBuilder private var content: some View {
        if Demo.streamAPIKey.isEmpty {
            ContentUnavailableView {
                Label("No Stream key yet", systemImage: "key")
            } description: {
                Text("Put the STREAM_API_KEY the router runs with in Demo.streamAPIKey.")
            }
        } else {
            TabView(selection: $tab) {
                ChatView(agent: Demo.agentName)
                    .tabItem { Label("Chat", systemImage: "text.bubble") }
                    .tag(Tab.chat)
                VoiceView(agent: Demo.agentName, style: voiceStyle)
                    .tabItem { Label("Voice", systemImage: "waveform") }
                    .tag(Tab.voice)
            }
        }
    }
}
