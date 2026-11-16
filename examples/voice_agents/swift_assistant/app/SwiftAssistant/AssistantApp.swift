import StreamChatAI
import SwiftUI

@main
struct AssistantApp: App {
    init() {
        // The composer floats over the conversation in glass of its own, which a fill would hide.
        let colors = AIAppearance.shared.colors
        colors.composerBackground = .clear
        colors.composerAttachmentButtonBackground = .clear
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}
