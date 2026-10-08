import Foundation

/// What went wrong talking to the router.
///
/// Cancellation is never one of these. A cancelled task throws `CancellationError`, which is
/// left alone so that a view that goes away mid-request is not reported as a failure.
public enum AgentsError: Error, Sendable {
    /// The router, or something in front of it, answered with a failure.
    case http(HTTPFailure)
    /// The request never got an answer.
    case transport(any Error)
    /// The socket ended before the session did.
    case socketClosed(code: Int, reason: String)
    /// The router answered with something this SDK cannot read.
    case unreadable(String)
    /// The SDK was set up in a way no request can go out under, like an API key with no user.
    case configuration(String)
}

/// A failure the router answered, as its body and headers tell it.
///
/// Every failure the router answers says all of this. A body that is not the router's — a
/// proxy's 502 page, an empty one, an older router's `{"error": "..."}` — keeps its text as
/// `message` and leaves `type`, `code` and `docURL` empty; the request id is read either way.
public struct HTTPFailure: Sendable, Hashable {
    public let status: Int
    /// The kind of failure, which decides the status: `invalid_request`, `authentication`,
    /// `permission`, `not_found`, `conflict`, `rate_limited`, `internal`, `unavailable` and a few
    /// more. A string, so a kind added to the router later still reads.
    public let type: String
    /// What went wrong, for a program to branch on: `not_configured`, `validation_failed`,
    /// `server_side_only`, `session_not_found` and more as the router learns them, so expect
    /// one you do not know.
    public let code: String
    /// What went wrong, for a person to read. Its wording may change; branch on `code`.
    public let message: String
    /// Where `code` is explained.
    public let docURL: String
    /// The response's `X-Request-Id`, which is what to quote to support: a 500 says only
    /// "something went wrong", and this is how the rest of it is found.
    public let requestID: String

    public init(
        status: Int, message: String, type: String = "", code: String = "", docURL: String = "",
        requestID: String = ""
    ) {
        self.status = status
        self.type = type
        self.code = code
        self.message = message
        self.docURL = docURL
        self.requestID = requestID
    }
}

/// How much of a failure's body is read. The router's envelope is a few hundred bytes; a proxy's
/// error page can be anything.
let maximumFailureBody = 64 << 10

/// How much of a body that is not the envelope becomes the message.
private let maximumFailureText = 1 << 10

extension HTTPFailure {
    /// What a failure's body says: the router's envelope, or else its text.
    init(status: Int, requestID: String, body: Data) {
        let detail =
            (try? JSONDecoder().decode(JSONValue.self, from: body))?.objectValue["error"]?
            .objectValue ?? [:]
        let message = detail["message"]?.stringValue ?? ""
        guard !message.isEmpty else {
            let text = String(decoding: body, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            self.init(
                status: status,
                message: text.isEmpty
                    ? HTTPURLResponse.localizedString(forStatusCode: status)
                    : String(text.prefix(maximumFailureText)),
                requestID: requestID)
            return
        }
        self.init(
            status: status, message: message, type: detail["type"]?.stringValue ?? "",
            code: detail["code"]?.stringValue ?? "", docURL: detail["doc_url"]?.stringValue ?? "",
            requestID: requestID)
    }
}

extension AgentsError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .http(let failure):
            return failure.requestID.isEmpty
                ? "the router answered \(failure.status): \(failure.message)"
                : "the router answered \(failure.status): \(failure.message) (request \(failure.requestID))"
        case .transport(let underlying):
            return "could not reach the router: \(underlying.localizedDescription)"
        case .socketClosed(let code, let reason):
            return reason.isEmpty
                ? "the session socket closed (\(code))"
                : "the session socket closed (\(code)): \(reason)"
        case .unreadable(let what):
            return "could not read the router's answer: \(what)"
        case .configuration(let what):
            return what
        }
    }
}

extension AgentsError {
    /// A 403 from the router, which is what a client-side caller gets for a path that only a
    /// backend may take.
    public var isServerSideOnly: Bool {
        if case .http(let failure) = self { return failure.status == 403 }
        return false
    }
}
