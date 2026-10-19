import Foundation
import Observation
import VisionAgentsCore

/// This person's conversations with the assistant, most recently active first.
///
/// The router keeps them: a session belongs to whoever opened it, so this lists this phone's
/// user's and nobody else's. It titles each one itself once there is something to go on, and
/// a conversation that ended carries on when it is next asked something, so there is nothing
/// to keep here but the list.
@MainActor
@Observable
final class Conversations {
    /// The conversations found, newest first.
    private(set) var items: [Session] = []

    /// What the list is narrowed to, or empty for all of them.
    var searchText = ""

    /// Why the list could not be read, or nil.
    private(set) var failure: String?

    /// Reads the list again, narrowed to `searchText` when there is one.
    func reload() async {
        do {
            let query = SessionQuery(limit: 50)
            let page =
                searchText.isEmpty
                ? try await Assistant.sessions.query(query)
                : try await Assistant.sessions.search(searchText, query)
            // Only the ones kept in writing can be opened again as a chat: a session that was
            // only ever a call, or incognito, has no channel to read back.
            items = page.items.filter { !$0.conversationID.isEmpty }
            failure = nil
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
        }
    }

    /// Deletes a conversation: its turns, and what it remembered, go with it.
    func delete(_ session: Session) async {
        items.removeAll { $0.id == session.id }
        do {
            try await Assistant.sessions.delete(session.id)
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
            await reload()
        }
    }
}

extension Session {
    /// What a conversation is called in the list: the router's title once it has given it one.
    var displayTitle: String {
        title.isEmpty ? "New conversation" : title
    }
}
