import SwiftUI
import VisionAgentsCore

/// Picks an agent, then talks to it in writing or out loud.
struct ContentView: View {
    @State private var configs: [AgentConfig] = []
    @State private var failure: String?
    @State private var isLoading = true

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Agents")
        }
        .task(load)
    }

    @ViewBuilder private var content: some View {
        if isLoading {
            ProgressView()
        } else if let failure {
            ContentUnavailableView {
                Label("Cannot reach the router", systemImage: "network.slash")
            } description: {
                Text(failure)
            } actions: {
                Button("Try again") { Task { await load() } }
            }
        } else if configs.isEmpty {
            ContentUnavailableView {
                Label("No agents yet", systemImage: "person.crop.circle.badge.questionmark")
            } description: {
                Text("Run `go run ./configure` in examples/agents/swift_demo first.")
            }
        } else {
            List(configs) { config in
                NavigationLink(value: config) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(config.name).font(.headline)
                        if !config.skills.isEmpty {
                            Text(config.skills.joined(separator: ", "))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationDestination(for: AgentConfig.self) { AgentView(config: $0) }
        }
    }

    @Sendable private func load() async {
        isLoading = true
        failure = nil
        do {
            configs = try await Demo.agents.agentConfigs()
        } catch is CancellationError {
            return
        } catch {
            failure = error.localizedDescription
        }
        isLoading = false
    }
}

/// One agent, two ways to talk to it.
struct AgentView: View {
    let config: AgentConfig

    var body: some View {
        TabView {
            ChatView(agent: config.name)
                .tabItem { Label("Chat", systemImage: "text.bubble") }
            VoiceView(agent: config.name)
                .tabItem { Label("Voice", systemImage: "waveform") }
        }
        .navigationTitle(config.name)
        .navigationBarTitleDisplayMode(.inline)
    }
}
