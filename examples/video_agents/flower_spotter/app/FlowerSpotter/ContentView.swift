import SwiftUI
import VisionAgentsCore

/// Live calls the flower_spotter Python process has started.
struct ContentView: View {
    @State private var calls: [CallRecord] = []
    @State private var failure: String?
    @State private var isLoading = true

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Flower spotter")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            Task { await load() }
                        } label: {
                            Image(systemName: "arrow.clockwise")
                        }
                    }
                }
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
                Text(
                    """
                    \(Demo.routerURL.absoluteString)
                    \(failure)
                    Set Demo.routerURL to `ipconfig getifaddr en0` on the Mac and rebuild. Allow local network access when asked.
                    """
                )
            } actions: {
                Button("Try again") { Task { await load() } }
            }
        } else if calls.isEmpty {
            ContentUnavailableView {
                Label("No live call", systemImage: "leaf")
            } description: {
                Text("Run `uv run flower_spotter.py run` in examples/video_agents/flower_spotter first.")
            } actions: {
                Button("Try again") { Task { await load() } }
            }
        } else {
            List(calls) { record in
                NavigationLink(value: record) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(record.callID).font(.headline)
                        Text(record.startedAt.formatted())
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationDestination(for: CallRecord.self) { CallView(record: $0) }
        }
    }

    @Sendable private func load() async {
        isLoading = true
        failure = nil
        defer { isLoading = false }
        do {
            let records = try await Demo.agents.calls(
                agentID: Demo.agentName, limit: 20)
            calls = records.filter(\.isRunning)
        } catch is CancellationError {
            return
        } catch {
            let ns = error as NSError
            failure = "\(error.localizedDescription) [\(ns.domain) \(ns.code)]"
        }
    }
}
