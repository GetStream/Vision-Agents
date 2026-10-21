import StreamChat
import StreamChatAI
import SwiftUI
import VisionAgentsCore

/// One message: the person's in a bubble with any photos they sent, the agent's as its steps
/// and then its answer.
struct MessageRow: View {
    let message: ChatMessage
    /// The photos the person sent with it from this phone, shown in place of Chat's copies.
    let photos: [UIImage]
    let reasoning: LiveReasoning
    let approver: AIToolApprover

    /// Whether the text is shown as still being written, which is what types it out.
    ///
    /// Stream folds a fast reply's live updates into one, so its text can arrive in the same
    /// update that says it is finished. `StreamingMessageView` types only what arrives while it
    /// is told the reply is being written, so it is told that until the text has reached it.
    @State private var writing: Bool

    init(message: ChatMessage, photos: [UIImage] = [], reasoning: LiveReasoning, approver: AIToolApprover) {
        self.message = message
        self.photos = photos
        self.reasoning = reasoning
        self.approver = approver
        _writing = State(initialValue: message.isGenerating)
    }

    var body: some View {
        if message.isAgentReply {
            VStack(alignment: .leading, spacing: 8) {
                let parts = message.parts
                if !parts.isEmpty {
                    AIMessagePartsView(parts: parts) { part in
                        // The step keeps only the opening of its thinking; the whole of it
                        // came with the live updates, while the reply was being written.
                        if let step = part.reasoning {
                            StreamingReasoningView(part: step, text: reasoning[message: message.id, step: step.id])
                        } else {
                            AIMessagePartView(part: part, approver: approver)
                        }
                    }
                }
                if !message.text.isEmpty {
                    StreamingMessageView(content: message.text, isGenerating: writing)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .task(id: message.isGenerating) {
                guard !message.isGenerating else {
                    writing = true
                    return
                }
                try? await Task.sleep(for: .milliseconds(100))
                writing = false
            }
        } else {
            VStack(alignment: .trailing, spacing: 6) {
                if photos.isEmpty {
                    ForEach(message.imageAttachments, id: \.id) { image in
                        AsyncImage(url: image.imageURL) { loaded in
                            loaded.resizable().scaledToFill()
                        } placeholder: {
                            Color.secondary.opacity(0.15)
                        }
                        .frame(width: 180, height: 180)
                        .clipShape(.rect(cornerRadius: 14))
                    }
                } else {
                    SentPhotos(photos: photos)
                }
                if !message.text.isEmpty {
                    PersonBubble(text: message.text)
                }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }
}

/// Photos the person sent from this phone, shown from it.
struct SentPhotos: View {
    let photos: [UIImage]

    var body: some View {
        ForEach(photos.indices, id: \.self) { index in
            Image(uiImage: photos[index])
                .resizable()
                .scaledToFill()
                .frame(width: 180, height: 180)
                .clipShape(.rect(cornerRadius: 14))
        }
    }
}

/// What the person said, in a bubble at the trailing edge.
struct PersonBubble: View {
    let text: String

    var body: some View {
        Text(text)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.tint.opacity(0.15), in: .rect(cornerRadius: 18))
            .frame(maxWidth: .infinity, alignment: .trailing)
    }
}
