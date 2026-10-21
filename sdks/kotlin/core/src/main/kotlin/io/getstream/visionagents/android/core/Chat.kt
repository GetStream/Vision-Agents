package io.getstream.visionagents.android.core

import android.content.Context
import io.getstream.chat.android.client.ChatClient
import io.getstream.chat.android.client.channel.ChannelClient
import io.getstream.chat.android.client.token.TokenProvider
import io.getstream.chat.android.models.Message
import io.getstream.chat.android.models.User
import io.getstream.result.Result
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.runBlocking

private const val KIND = "chat"

/** The last client built here, so the process-wide instance can be told apart from the app's. */
@Volatile private var builtHere: ChatClient? = null

/**
 * Hands over the app's own Stream Chat client, so every conversation opens on it rather than on
 * a second one. Connecting it, as the user `setUser` named, stays the app's job, and
 * [VisionAgents.disconnect] leaves it alone.
 */
public fun VisionAgents.use(client: ChatClient) {
    backend.give(KIND, client)
}

/**
 * The Stream Chat channel this conversation is kept in.
 *
 * It opens on the client handed over with [use], or else on one built here and connected as the
 * user `setUser` named, with the agents' key, which every session then shares. `ChatClient` is
 * one per process, so one the app built without handing it over is used if it is connected as
 * the same user and refused otherwise, rather than replaced.
 */
public suspend fun AgentSession.chat(context: Context): ChannelClient {
    val channel = session.conversationId
    if (channel.isEmpty()) {
        throw AgentsException.Configuration(
            "this session keeps no transcript, so there is no channel to read: only a text " +
                "session has one, and an incognito session never does",
        )
    }
    val user = backend.user
    if (ChatClient.isInitialized && ChatClient.instance() !== builtHere) {
        adopt(backend, ChatClient.instance())
    }
    val client = backend.shared(
        KIND,
        open = { connect(context, backend, it.apiKey, it.user, it.token) },
        disconnect = { it.disconnect(flushPersistence = false).await() },
    )
    val connected = client.getCurrentUser()?.id
    if (connected != null && user != null && connected != user.id) {
        throw AgentsException.Configuration(
            "the chat client is connected as $connected but setUser named ${user.id}: connect " +
                "it as the same user, or the conversation is somebody else's",
        )
    }
    return client.channel(channel)
}

/** Uses a client the app built without handing it over, if it is connected as the same user. */
private fun adopt(backend: Backend, existing: ChatClient) {
    val connected = existing.getCurrentUser()?.id
    if (connected == null || connected != backend.user?.id) {
        throw AgentsException.Configuration(
            "this process already has a ChatClient, connected as ${connected ?: "nobody"}: " +
                "connect it as the user setUser named and hand it over with agents.use(client), " +
                "since building another would replace it",
        )
    }
    backend.give(KIND, existing)
}

/** A client connected as the agents' user, asking them for a fresh token when Stream says it expired. */
private suspend fun connect(
    context: Context,
    backend: Backend,
    apiKey: String,
    user: io.getstream.visionagents.android.core.User,
    token: String,
): ChatClient {
    val client = ChatClient.Builder(apiKey, context.applicationContext).build()
    builtHere = client
    val unused = AtomicReference<String?>(token)
    // Stream calls this on a worker thread of its own, and waits for the answer.
    val tokens = object : TokenProvider {
        override fun loadToken(): String =
            unused.getAndSet(null) ?: runBlocking { backend.streamCredentials(refresh = true).token }
    }
    val connecting = client.connectUser(
        User(id = user.id, name = user.name, image = user.image),
        tokens,
    ).await()
    if (connecting is Result.Failure) throw AgentsException.ChatFailed(connecting.value.message)
    return client
}

/** Reads the window of thinking a live update of a reply carries, if it carries one. */
public fun LiveReasoning.read(message: Message) {
    val window = message.extraData["reasoning"] as? Map<*, *> ?: return
    val id = window["id"] as? String ?: return
    val offset = window["offset"] as? Number ?: return
    val text = window["text"] as? String ?: return
    add(text, offset.toInt(), message.id, id)
}

