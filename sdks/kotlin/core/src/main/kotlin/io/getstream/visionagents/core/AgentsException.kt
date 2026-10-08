package io.getstream.visionagents.core

/**
 * What went wrong talking to the router.
 *
 * Cancellation is never one of these. A cancelled coroutine throws `CancellationException`,
 * which is left alone so that a screen that goes away mid-request is not reported as a failure.
 */
public sealed class AgentsException(message: String, cause: Throwable? = null) :
    Exception(message, cause) {

    /**
     * The router refused the request, or the socket's handshake. [reason] is what it said.
     *
     * [type] is the kind of failure, which decides the status (`invalid_request`, `permission`,
     * `not_found`, `rate_limited`, ...), and [code] is what to branch on (`validation_failed`,
     * `session_not_found`, ...); both are strings, because the router adds values. [docUrl]
     * explains the code. All three are null when the body was not the router's, such as a
     * proxy's page, and [reason] is then that body, cut short, or the status phrase.
     *
     * [requestId] is the response's `X-Request-Id`, which is what to quote to support: a 500
     * says only "something went wrong". [retryAfterSeconds] is set on a 429, which is a device
     * that has used up its day.
     */
    public class Http(
        public val status: Int,
        public val reason: String,
        public val retryAfterSeconds: Long? = null,
        public val type: String? = null,
        public val code: String? = null,
        public val docUrl: String? = null,
        public val requestId: String? = null,
    ) : AgentsException(
        "the router answered $status: $reason" + requestId?.let { " (request $it)" }.orEmpty(),
    )

    /** The request never got an answer. */
    public class Transport(cause: Throwable) :
        AgentsException("could not reach the router: ${cause.message}", cause)

    /** The socket ended before the session did. */
    public class SocketClosed(public val code: Int, public val reason: String) :
        AgentsException(
            if (reason.isEmpty()) "the session socket closed ($code)"
            else "the session socket closed ($code): $reason",
        )

    /** The router answered with something this SDK cannot read. */
    public class Unreadable(what: String, cause: Throwable? = null) :
        AgentsException("could not read the router's answer: $what", cause)

    /** What was asked for cannot be sent, and was refused before any request. */
    public class Configuration(message: String) : AgentsException(message)

    /**
     * A 403 from the router, which is what a device gets for a path only a backend may take,
     * and what an app that turned guests away answers a guest.
     */
    public val isServerSideOnly: Boolean
        get() = this is Http && status == 403
}
