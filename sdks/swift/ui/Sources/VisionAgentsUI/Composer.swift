import SwiftUI

/// Where you type.
///
/// Sending is a closure rather than a binding onto a session, so the same field works for a
/// conversation, a search box or anything else the host wants it for.
public struct Composer: View {
    private let prompt: String
    private let isEnabled: Bool
    private let send: (String) async -> Void

    @State private var text = ""
    @FocusState private var isFocused: Bool

    public init(
        prompt: String = "Message",
        isEnabled: Bool = true,
        send: @escaping (String) async -> Void
    ) {
        self.prompt = prompt
        self.isEnabled = isEnabled
        self.send = send
    }

    public var body: some View {
        HStack(spacing: 8) {
            TextField(prompt, text: $text, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .focused($isFocused)
                .onSubmit(submit)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.quinary, in: .capsule)

            Button(action: submit) {
                Image(systemName: "arrow.up")
                    .font(.body.weight(.semibold))
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.borderedProminent)
            .clipShape(.circle)
            .disabled(!canSend)
        }
        .disabled(!isEnabled)
    }

    private var canSend: Bool {
        isEnabled && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func submit() {
        guard canSend else { return }
        let sending = text
        text = ""
        Task { await send(sending) }
    }
}
