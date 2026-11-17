import StreamChatAI
import SwiftUI

/// The conversation in front, and the ones there were in a sidebar that slides over it.
struct RootView: View {
    @State private var conversations = Conversations()
    @State private var isMenuOpen = false
    /// The conversation open, or nil for a new one.
    @State private var selected: String?
    /// Which conversation the screen was opened for. It changes only when the person picks
    /// one, so a new conversation getting its session keeps the screen it was started on.
    @State private var screen = UUID()

    var body: some View {
        if Assistant.streamAPIKey.isEmpty {
            ContentUnavailableView {
                Label("No Stream key yet", systemImage: "key")
            } description: {
                Text("Put the STREAM_API_KEY the router runs with in Assistant.streamAPIKey.")
            }
        } else {
            conversation
        }
    }

    private var conversation: some View {
        SidebarView(isOpen: $isMenuOpen, excludedBottomHeight: 80) {
            ConversationList(
                conversations: conversations, selected: selected,
                onNew: { show(nil) }, onOpen: { show($0.id) })
        } content: {
            NavigationStack {
                ChatScreen(sessionID: selected) { started in
                    selected = started
                }
                .id(screen)
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            isMenuOpen.toggle()
                        } label: {
                            Image(systemName: "line.3.horizontal")
                        }
                        .accessibilityLabel("Conversations")
                    }
                }
            }
        }
        .onChange(of: isMenuOpen) { _, isOpen in
            // The router titles a conversation as it goes, so the list is read again each
            // time it is shown. The sidebar slides over the composer without ending its
            // editing, so the keyboard is put away here.
            guard isOpen else { return }
            UIApplication.shared.sendAction(
                #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            Task { await conversations.reload() }
        }
    }

    private var title: String {
        conversations.items.first { $0.id == selected }?.displayTitle ?? "Juno"
    }

    private func show(_ sessionID: String?) {
        selected = sessionID
        screen = UUID()
        isMenuOpen = false
    }
}
