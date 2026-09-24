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
     * The router refused the request. [reason] is what it said, not a status phrase.
     *
     * [retryAfterSeconds] is set on a 429, which is a device that has used up its day.
     */
    public class Http(
        public val status: Int,
        public val reason: String,
        public val retryAfterSeconds: Long? = null,
    ) : AgentsException("the router answered $status: $reason")

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
