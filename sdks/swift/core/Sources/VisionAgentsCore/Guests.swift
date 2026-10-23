import Foundation
import StreamCore

/// Somewhere to keep a guest between launches.
///
/// Narrow on purpose, so an app can put it wherever it keeps things: the keychain,
/// `UserDefaults`, or `FileGuestStore`.
public protocol GuestStore: Sendable {
    func read() -> String?
    func write(_ value: String) throws
    func clear() throws
}

/// A guest kept in one file, such as `vision-agents-guest.json` in Application Support.
///
/// What it holds is a token scoped to one guest, so the app's own container is enough.
public struct FileGuestStore: GuestStore {
    public let url: URL

    public init(url: URL) {
        self.url = url
    }

    public func read() -> String? {
        try? String(contentsOf: url, encoding: .utf8)
    }

    public func write(_ value: String) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(value.utf8).write(to: url, options: .atomic)
    }

    public func clear() throws {
        do {
            try FileManager.default.removeItem(at: url)
        } catch CocoaError.fileNoSuchFile {
            // Nothing remembered is already forgotten.
        }
    }
}

/// What a device asking for a guest with no token is told when the router, or Stream's proxy in
/// front of it, admits no anonymous callers.
let anonymousGuestRefused =
    "this router admits no anonymous callers, so a device cannot mint its own guest: mint it "
    + "on your backend with the server SDK's guestUser and pass it to setUser"

/// Who a new guest is, for a transcript a person reads later.
public struct GuestOptions: Sendable {
    public var name: String?
    public var custom: [String: RawJSON]?
    /// Mint a new guest even if one is remembered. For a "not me" button: the person holding
    /// the phone is somebody else, and the remembered guest would hand them the last one's
    /// conversations.
    public var fresh: Bool

    public init(name: String? = nil, custom: [String: RawJSON]? = nil, fresh: Bool = false) {
        self.name = name
        self.custom = custom
        self.fresh = fresh
    }
}

extension Backend {
    /// Gets the remembered guest, or creates one.
    ///
    /// A remembered guest whose token has expired is asked for again by id, which the router
    /// answers with the same guest and a fresh token: coming back is the same person.
    func guest(_ options: GuestOptions, store: (any GuestStore)?) async throws -> Guest {
        let held = options.fresh ? nil : store?.read().flatMap(remembered)
        if let held, held.expiresAt.map({ $0 > Date() }) ?? true {
            return held
        }

        let body = GuestUserRequest(custom: options.custom, id: held?.id, name: options.name)
        let guest = Guest(try await call { try await $0.createGuestUser(guestUserRequest: body) })
        try store?.write(remembering(guest))
        return guest
    }
}

/// A guest as a store keeps it, in the shape the router sent it.
private struct Remembered: Codable {
    let id: String
    let token: String
    let name: String?
    let custom: [String: RawJSON]?
    let expiresAt: String?

    enum CodingKeys: String, CodingKey {
        case id, token, name, custom
        case expiresAt = "expires_at"
    }
}

private func remembering(_ guest: Guest) -> String {
    let value = Remembered(
        id: guest.id, token: guest.token, name: guest.name.isEmpty ? nil : guest.name,
        custom: guest.custom.isEmpty ? nil : guest.custom,
        expiresAt: guest.expiresAt.map { RouterDates().encode($0) })
    return String(decoding: (try? JSONEncoder().encode(value)) ?? Data(), as: UTF8.self)
}

/// What the store held, or nil when it held nothing usable. Something else under the key, or a
/// truncated write, is answered by minting a new guest rather than by being unable to ask anything.
private func remembered(_ value: String) -> Guest? {
    guard let held = try? JSONDecoder().decode(Remembered.self, from: Data(value.utf8)),
        !held.id.isEmpty, !held.token.isEmpty
    else { return nil }
    return Guest(
        id: held.id, token: held.token, name: held.name ?? "", custom: held.custom ?? [:],
        expiresAt: held.expiresAt.flatMap { try? RouterDates().decode($0) })
}
