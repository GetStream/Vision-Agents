import StreamChatAI
import SwiftUI

/// Where you type.
///
/// Sending is a closure rather than a binding onto a session, so the same field works for a
/// conversation, a search box or anything else the host wants it for.
///
/// Dictation is part of this field, so the host app needs `NSMicrophoneUsageDescription` and
/// `NSSpeechRecognitionUsageDescription` in its Info.plist. A package cannot supply either,
/// and iOS terminates an app that asks for one it has not declared. It records through
/// `AVAudioSession` too, so do not put this on screen during a call: the call owns the audio
/// session.
public struct Composer: View {
    private let isEnabled: Bool
    private let isGenerating: Bool
    private let send: (String) async -> Void
    private let stop: (() async -> Void)?

    /// - Parameters:
    ///   - isEnabled: whether there is anywhere to send to.
    ///   - isGenerating: whether a reply is in flight, which swaps send for stop.
    ///   - send: what to do with what was typed.
    ///   - stop: how to abandon the reply in flight. Omit it to leave the reply alone.
    public init(
        isEnabled: Bool = true,
        isGenerating: Bool = false,
        send: @escaping (String) async -> Void,
        stop: (() async -> Void)? = nil
    ) {
        self.isEnabled = isEnabled
        self.isGenerating = isGenerating
        self.send = send
        self.stop = stop
    }

    public var body: some View {
        ComposerView(
            viewFactory: TextOnlyComposerFactory(),
            isGenerating: isGenerating,
            onMessageSend: { message in
                let text = message.text
                Task { await send(text) }
            },
            onStopGenerating: stop.map { stop in { Task { await stop() } } }
        )
        .disabled(!isEnabled)
    }
}

/// The composer without its attachment button.
///
/// A session carries text, so photos picked here would have nowhere to go, and reaching the
/// photo library at all needs usage strings this package cannot supply. Dropping the leading
/// slot drops the picker sheet with it, since nothing opens it.
private struct TextOnlyComposerFactory: ComposerViewFactory {
    func makeLeadingComposerView(options: LeadingComposerViewOptions) -> some View {
        EmptyView()
    }

    func makeComposerPickerView(options: ComposerPickerViewOptions) -> some View {
        EmptyView()
    }
}
