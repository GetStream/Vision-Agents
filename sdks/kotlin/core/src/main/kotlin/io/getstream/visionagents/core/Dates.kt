package io.getstream.visionagents.core

import kotlin.time.Instant

/**
 * Reads a timestamp the router wrote.
 *
 * Go's `time.Time` marshals to RFC 3339 with however many fractional digits the value needs
 * and none when it needs none, so one is `...:50.89279Z` and the next `...:50Z`. The generated
 * models keep them as strings so that reading them is decided here, once.
 */
internal fun instant(value: String): Instant = try {
    Instant.parse(value)
} catch (e: IllegalArgumentException) {
    throw AgentsException.Unreadable("$value is not a timestamp", e)
}

internal fun instantOrNull(value: String?): Instant? =
    if (value.isNullOrEmpty()) null else instant(value)
