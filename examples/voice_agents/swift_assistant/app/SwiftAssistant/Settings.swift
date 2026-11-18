import SwiftUI

/// How the agent answers: in Markdown, or with interactive UI in A2UI or OpenUI.
///
/// The backend keeps it, as what the agent is told, so the app has nothing to switch: its
/// conversations show A2UI and OpenUI alike, and a reply without UI as its text.
enum ReplyFormat: String, CaseIterable, Identifiable, Codable {
    case markdown
    case a2ui
    case openui

    var id: Self { self }

    var title: String {
        switch self {
        case .markdown: "Markdown"
        case .a2ui: "A2UI"
        case .openui: "OpenUI"
        }
    }

    /// The format the agent answers in.
    static func current() async throws -> ReplyFormat {
        try await asked(URLRequest(url: Assistant.backendURL.appending(path: "reply-format")))
    }

    /// Has the agent answer in this format, from its next conversation on.
    func choose() async throws {
        var request = URLRequest(url: Assistant.backendURL.appending(path: "reply-format"))
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(Choice(format: self))
        _ = try await Self.asked(request)
    }

    private struct Choice: Codable {
        let format: ReplyFormat
    }

    private static func asked(_ request: URLRequest) async throws -> ReplyFormat {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw BackendError(said: String(decoding: data, as: UTF8.self))
        }
        return try JSONDecoder().decode(Choice.self, from: data).format
    }
}

/// What the backend said instead of doing what it was asked.
struct BackendError: LocalizedError {
    let said: String

    var errorDescription: String? {
        let said = said.trimmingCharacters(in: .whitespacesAndNewlines)
        return said.isEmpty ? "The backend didn't answer." : said
    }
}

/// The app's settings: how the agent answers, and the brand its UI wears.
struct SettingsScreen: View {
    /// Where whether this device's new conversations may have UI is kept.
    static let cardsKey = "cardsInNewConversations"

    @Environment(\.dismiss) private var dismiss
    @AppStorage(Brand.storageKey) private var brand = Brand.stream
    @AppStorage(SettingsScreen.cardsKey) private var cards = true
    @State private var format: ReplyFormat?
    @State private var isChoosing = false
    @State private var failure: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if let format {
                        Picker("Replies", selection: Binding(get: { format }, set: choose)) {
                            ForEach(ReplyFormat.allCases) { format in
                                Text(format.title).tag(format)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(isChoosing)
                    } else if failure == nil {
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    }
                } header: {
                    Text("Replies")
                } footer: {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("""
                        Markdown answers in text. A2UI and OpenUI answer plans and bookings with \
                        interactive UI, made of the app's own views: maps, restaurant cards, gauges \
                        and forms. New conversations answer this way, and earlier replies keep theirs.
                        """)
                        if let failure {
                            Text(failure)
                                .foregroundStyle(.red)
                        }
                    }
                }
                Section {
                    Toggle("Interactive UI", isOn: $cards)
                } footer: {
                    Text("""
                    Off leaves the UI instructions the replies above chose out of this device's \
                    new conversations, so the agent is expected to answer in text.
                    """)
                }
                Section {
                    ForEach(Brand.allCases) { choice in
                        Button {
                            brand = choice
                        } label: {
                            HStack(spacing: 12) {
                                RoundedRectangle(cornerRadius: choice.cornerRadius / 2)
                                    .fill(choice.accent)
                                    .frame(width: 28, height: 28)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(choice.title)
                                        .foregroundStyle(.primary)
                                    Text(choice.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if choice == brand {
                                    Image(systemName: "checkmark")
                                        .fontWeight(.semibold)
                                        .foregroundStyle(choice.accent)
                                }
                            }
                        }
                        .accessibilityAddTraits(choice == brand ? .isSelected : [])
                    }
                } header: {
                    Text("Brand")
                } footer: {
                    Text("The accent and corners the agent's UI wears, in every conversation, straight away.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .task { await load() }
        }
    }

    private func load() async {
        do {
            format = try await ReplyFormat.current()
        } catch {
            failure = error.localizedDescription
        }
    }

    private func choose(_ chosen: ReplyFormat) {
        let previous = format
        format = chosen
        isChoosing = true
        Task {
            do {
                try await chosen.choose()
                failure = nil
            } catch {
                format = previous
                failure = error.localizedDescription
            }
            isChoosing = false
        }
    }
}
