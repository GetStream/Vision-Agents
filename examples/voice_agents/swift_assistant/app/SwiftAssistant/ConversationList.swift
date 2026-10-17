import SwiftUI
import VisionAgentsCore

/// The sidebar: a new conversation, a search, and the conversations there were, by day.
struct ConversationList: View {
    @Bindable var conversations: Conversations
    /// The conversation open beside this, or nil for a new one.
    let selected: String?
    let onNew: () -> Void
    let onOpen: (Session) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: onNew) {
                Label("New chat", systemImage: "square.and.pencil")
                    .font(.headline)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .buttonStyle(.plain)

            TextField("Search conversations", text: $conversations.searchText)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)
                .submitLabel(.search)

            List {
                ForEach(sections, id: \.id) { section in
                    Section(section.title) {
                        ForEach(section.items) { session in
                            Button {
                                onOpen(session)
                            } label: {
                                Text(session.displayTitle)
                                    .lineLimit(1)
                                    .fontWeight(session.id == selected ? .semibold : .regular)
                            }
                            .swipeActions {
                                Button("Delete", role: .destructive) {
                                    Task { await conversations.delete(session) }
                                }
                            }
                        }
                    }
                }
                if let failure = conversations.failure {
                    Text(failure)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .refreshable { await conversations.reload() }
        }
        .task(id: conversations.searchText) {
            // Waits for typing to pause before searching.
            if !conversations.searchText.isEmpty {
                try? await Task.sleep(for: .milliseconds(300))
                guard !Task.isCancelled else { return }
            }
            await conversations.reload()
        }
    }

    private struct Day {
        let title: String
        let items: [Session]

        /// A day can appear twice: the list is in order of activity, and the days are when
        /// each conversation started, which is all a `Session` here says.
        var id: String { items.first?.id ?? title }
    }

    /// The conversations under Today, Yesterday and then their week, in the router's order.
    private var sections: [Day] {
        var days: [Day] = []
        for session in conversations.items {
            let title = Self.title(for: session.createdAt)
            if days.last?.title == title {
                days[days.count - 1] = Day(title: title, items: days[days.count - 1].items + [session])
            } else {
                days.append(Day(title: title, items: [session]))
            }
        }
        return days
    }

    private static func title(for date: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(date) { return "Today" }
        if calendar.isDateInYesterday(date) { return "Yesterday" }
        if let days = calendar.dateComponents([.day], from: date, to: .now).day, days < 7 {
            return "This week"
        }
        return date.formatted(.dateTime.month(.wide).year())
    }
}
