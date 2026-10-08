package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.GuestUser as GuestSchema
import io.getstream.visionagents.core.generated.GuestUserRequest
import java.io.File
import java.io.IOException
import kotlin.time.Clock
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.JsonObject

/**
 * Somewhere to keep a guest between launches.
 *
 * Narrow on purpose, so an app can put it wherever it keeps things: DataStore, encrypted
 * preferences, or [FileGuestStore].
 */
public interface GuestStore {
    public fun read(): String?

    public fun write(value: String)

    public fun clear()
}

/**
 * A guest kept in one file, such as `File(context.filesDir, "vision-agents-guest.json")`.
 *
 * What it holds is a token scoped to one guest, so app-private storage is enough.
 */
public class FileGuestStore(private val file: File) : GuestStore {
    override fun read(): String? = try {
        file.readText()
    } catch (_: IOException) {
        null
    }

    override fun write(value: String) {
        file.parentFile?.mkdirs()
        file.writeText(value)
    }

    override fun clear() {
        file.delete()
    }
}

/** Who a new guest is, for a transcript a person reads later. */
public data class GuestOptions(
    val name: String? = null,
    val custom: JsonObject? = null,
    /**
     * Mint a new guest even if one is remembered. For a "not me" button: the person holding
     * the phone is somebody else, and the remembered guest would hand them the last one's
     * conversations.
     */
    val fresh: Boolean = false,
)

/**
 * Gets the remembered guest, or creates one.
 *
 * A remembered guest whose token has expired is asked for again by id, which the router answers
 * with the same guest and a fresh token: coming back is the same person.
 */
internal suspend fun Backend.guest(options: GuestOptions, store: GuestStore?): Guest {
    val held = if (options.fresh) null else store?.read()?.let(::remembered)
    if (held != null) {
        val expiry = instantOrNull(held.expiresAt)
        if (expiry == null || expiry > Clock.System.now()) return Guest.of(held)
    }

    val minted = post(
        listOf("v1", "agents", "guests"),
        GuestUserRequest.serializer(),
        GuestUserRequest(id = held?.id, name = options.name, custom = options.custom),
        GuestSchema.serializer(),
    )
    store?.write(wire.encodeToString(GuestSchema.serializer(), minted))
    return Guest.of(minted)
}

/**
 * What the store held, or null when it held nothing usable. Something else under the key, or a
 * truncated write, is answered by minting a new guest rather than by being unable to ask anything.
 */
private fun remembered(value: String): GuestSchema? = try {
    wire.decodeFromString(GuestSchema.serializer(), value).takeIf { it.id.isNotEmpty() && it.token.isNotEmpty() }
} catch (_: SerializationException) {
    null
} catch (_: IllegalArgumentException) {
    null
}
