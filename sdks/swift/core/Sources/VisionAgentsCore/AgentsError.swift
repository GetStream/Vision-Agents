import Foundation

/// What went wrong talking to the router.
///
/// Cancellation is never one of these. A cancelled task throws `CancellationError`, which is
/// left alone so that a view that goes away mid-request is not reported as a failure.
public enum AgentsError: Error, Sendable {
    /// The router refused the request. `message` is what it said, not a status phrase.
    case http(status: Int, message: String)
    /// The request never got an answer.
    case transport(any Error)
    /// The socket ended before the session did.
    case socketClosed(code: Int, reason: String)
    /// No agent config of that name belongs to this customer.
    case unknownAgent(String)
    /// The router answered with something this SDK cannot read.
    case unreadable(String)
}

extension AgentsError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .http(let status, let message):
            return "the router answered \(status): \(message)"
        case .transport(let underlying):
            return "could not reach the router: \(underlying.localizedDescription)"
        case .socketClosed(let code, let reason):
            return reason.isEmpty
                ? "the session socket closed (\(code))"
                : "the session socket closed (\(code)): \(reason)"
        case .unknownAgent(let name):
            return "no agent config called \(name)"
        case .unreadable(let what):
            return "could not read the router's answer: \(what)"
        }
    }
}

extension AgentsError {
    /// A 403 from the router, which is what a client-side caller gets for a path that only a
    /// backend may take.
    public var isServerSideOnly: Bool {
        if case .http(403, _) = self { return true }
        return false
    }
}
