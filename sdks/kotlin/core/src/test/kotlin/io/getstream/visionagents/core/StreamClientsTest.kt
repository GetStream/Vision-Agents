package io.getstream.visionagents.core

import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest

/** Stands in for a Stream client: something with an identity of its own, connected as a user. */
private class Peer(val user: String) {
    var connected = true
}

/** What a chat or video module opens and closes: every peer it built, and whether it is still connected. */
private class Peers {
    val built: MutableList<Peer> = Collections.synchronizedList(mutableListOf())

    val open: suspend (StreamCredentials) -> Peer = { Peer(it.user.id).also(built::add) }
    val disconnect: suspend (Peer) -> Unit = { it.connected = false }
}

@OptIn(StreamClients::class)
class StreamClientsTest {
    private fun signedIn(user: String): Backend =
        Backend(apiKey = "key").apply { setUser(User(user), "token-for-$user") }

    @Test
    fun `the token is asked for once, and again only on a refresh`() = runTest {
        val asked = AtomicInteger()
        val backend = Backend(apiKey = "key")
        backend.setUser(User("john", "John")) { "token-${asked.incrementAndGet()}" }

        val first = backend.streamCredentials()
        val again = backend.streamCredentials()
        val refreshed = backend.streamCredentials(refresh = true)

        assertEquals(StreamCredentials("key", User("john", "John"), "token-1"), first)
        assertEquals("token-1", again.token)
        assertEquals("token-2", refreshed.token)
    }

    @Test
    fun `two clients refreshing at once fetch one token`() = runTest {
        val asked = AtomicInteger()
        val backend = Backend(apiKey = "key")
        backend.setUser(User("john")) {
            val count = asked.incrementAndGet()
            delay(50)
            "token-$count"
        }
        backend.streamCredentials()

        val chat = async { backend.streamCredentials(refresh = true) }
        val video = async { backend.streamCredentials(refresh = true) }

        assertEquals(listOf("token-2", "token-2"), listOf(chat.await().token, video.await().token))
        assertEquals(2, asked.get())
    }

    @Test
    fun `refuses without a Stream key or a user`() = runTest {
        assertFailsWith<AgentsException.Configuration> { Backend(apiKey = "key").streamCredentials() }

        val local = Backend(url = "http://localhost:8080", customerId = "acme")
        local.setUser(User("john"), "token")
        assertFailsWith<AgentsException.Configuration> { local.streamCredentials() }
    }

    @Test
    fun `beside a customer id the key is Stream's`() = runTest {
        val local = Backend(url = "http://localhost:8080", customerId = "acme", apiKey = "stream-key")
        local.setUser(User("john"), "token-for-john")

        assertEquals(StreamCredentials("stream-key", User("john"), "token-for-john"), local.streamCredentials())
    }

    @Test
    fun `sessions asking at once share one client`() = runTest {
        val peers = Peers()
        val backend = signedIn("john")

        val asking = List(3) { async { backend.shared("chat", peers.open, peers.disconnect) } }
        val clients = asking.map { it.await() }

        assertEquals(1, peers.built.size)
        assertTrue(clients.all { it === clients.first() })
    }

    @Test
    fun `chat and video are a client each`() = runTest {
        val peers = Peers()
        val backend = signedIn("john")

        val chat = backend.shared("chat", peers.open, peers.disconnect)
        val video = backend.shared("video", peers.open, peers.disconnect)

        assertNotSame(chat, video)
        assertEquals(2, peers.built.size)
    }

    @Test
    fun `a client that failed to open is opened again`() = runTest {
        val tries = AtomicInteger()
        val backend = signedIn("john")
        val open: suspend (StreamCredentials) -> Peer = {
            if (tries.incrementAndGet() == 1) throw IllegalStateException("offline")
            Peer(it.user.id)
        }

        assertFailsWith<IllegalStateException> { backend.shared("chat", open) {} }
        val peer = backend.shared("chat", open) {}

        assertEquals("john", peer.user)
        assertEquals(2, tries.get())
    }

    @Test
    fun `another user gets a client of their own`() = runTest {
        val peers = Peers()
        val backend = signedIn("john")

        val john = backend.shared("chat", peers.open, peers.disconnect)
        backend.setUser(User("jane"), "token-for-jane")
        val jane = backend.shared("chat", peers.open, peers.disconnect)

        assertEquals("john", john.user)
        assertEquals("jane", jane.user)
    }

    @Test
    fun `the app's own client is used, kept, and left connected`() = runTest {
        val peers = Peers()
        val backend = signedIn("john")
        val mine = Peer("john")
        backend.give("chat", mine)

        val first = backend.shared("chat", peers.open, peers.disconnect)
        backend.setUser(User("john"), "another-token")
        val second = backend.shared("chat", peers.open, peers.disconnect)
        VisionAgents(backend).disconnect()

        assertSame(mine, first)
        assertSame(mine, second)
        assertTrue(peers.built.isEmpty())
        assertTrue(mine.connected)
    }

    @Test
    fun `disconnecting closes what was built, and the next session opens anew`() = runTest {
        val peers = Peers()
        val backend = signedIn("john")

        val first = backend.shared("chat", peers.open, peers.disconnect)
        VisionAgents(backend).disconnect()
        val second = backend.shared("chat", peers.open, peers.disconnect)

        assertNotSame(first, second)
        assertEquals(listOf(false, true), peers.built.map { it.connected })
    }
}
