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

/// A guest kept in one file, such as `stream-agents-guest.json` in Application Support.
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
    /// A remembered guest whose token has expired comes back with their id and the refresh
    /// secret they were last handed, which the router answers with the same guest and a fresh
    /// token. Without the secret, or with one already spent, the router mints somebody new,
    /// since a guest id is no proof of anything. Name and custom go along either way: a
    /// returning guest keeps their own, and a new one is given these. Whatever comes back
    /// replaces what was remembered, its next secret included.
    func guest(_ options: GuestOptions, store: (any GuestStore)?) async throws -> Guest {
        let held = options.fresh ? nil : store?.read().flatMap(remembered)
        if let held, held.guest.expiresAt.map({ $0 > Date() }) ?? true {
            return held.guest
        }

        // An id with no secret proves nothing and is ignored, so it is only sent with one.
        let secret = held?.refreshSecret
        let body = GuestUserRequest(
            custom: options.custom, id: secret == nil ? nil : held?.guest.id, name: options.name,
            refreshSecret: secret)
        let minted = try await call { try await $0.createGuestUser(guestUserRequest: body) }
        try store?.write(remembering(minted))
        return Guest(minted)
    }
}

/// A guest as a store keeps it, in the shape the router sent it.
///
/// The refresh secret is kept here and nowhere else: it is a credential for coming back as
/// this guest, and nothing an app does with a `Guest` needs it.
private struct Remembered: Codable {
    let id: String
    let token: String
    let name: String?
    let custom: [String: RawJSON]?
    let expiresAt: String?
    let refreshSecret: String?

    enum CodingKeys: String, CodingKey {
        case id, token, name, custom
        case expiresAt = "expires_at"
        case refreshSecret = "refresh_secret"
    }
}

private func remembering(_ guest: GuestUser) -> String {
    let value = Remembered(
        id: guest.id, token: guest.token, name: guest.name.flatMap { $0.isEmpty ? nil : $0 },
        custom: guest.custom.flatMap { $0.isEmpty ? nil : $0 },
        expiresAt: guest.expiresAt.map { RouterDates().encode($0) },
        refreshSecret: guest.refreshSecret)
    return String(decoding: (try? JSONEncoder().encode(value)) ?? Data(), as: UTF8.self)
}

/// What the store held, or nil when it held nothing usable. Something else under the key, or a
/// truncated write, is answered by minting a new guest rather than by being unable to ask anything.
private func remembered(_ value: String) -> (guest: Guest, refreshSecret: String?)? {
    guard let held = try? JSONDecoder().decode(Remembered.self, from: Data(value.utf8)),
        !held.id.isEmpty, !held.token.isEmpty
    else { return nil }
    let guest = Guest(
        id: held.id, token: held.token, name: held.name ?? "", custom: held.custom ?? [:],
        expiresAt: held.expiresAt.flatMap { try? RouterDates().decode($0) })
    return (guest, held.refreshSecret.flatMap { $0.isEmpty ? nil : $0 })
}
