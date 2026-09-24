package io.getstream.visionagents.core

import okhttp3.OkHttpClient

/**
 * The router, as a phone sees it.
 *
 * Two lines get a conversation going:
 *
 *     val agents = VisionAgents(url = "https://your-router", customerId = "acme")
 *     val chat = agents.agent("docs").chat()
 *
 * Opening a conversation, reading back its turns, going back to one of them, branching off and
 * ending it is the whole of what is here, because it is the whole of what the router lets a
 * device do. Configuring an agent and minting a token to join a call are server-side only: they
 * belong to a backend, which has the Go, Python or Node SDK and hands down what the app needs.
 */
public class VisionAgents(public val backend: Backend) : AutoCloseable {
    public constructor(
        url: String,
        customerId: String = "",
        apiKey: String = "",
        okHttpClient: OkHttpClient? = null,
    ) : this(Backend(url, customerId, apiKey, okHttpClient))

    /** Every conversation this caller has, whichever agent holds it. */
    public val sessions: Sessions = Sessions(backend)

    /** Says who this device is acting for, and how to prove it. See [Backend.setUser]. */
    public fun setUser(user: User, token: TokenProvider): Unit = backend.setUser(user, token)

    public fun setUser(user: User, token: String): Unit = backend.setUser(user, token)

    /** Acts as this guest from here on. */
    public fun setUser(guest: Guest): Unit = backend.setUser(guest)

    /** One agent, by the name its config was synced under. */
    public fun agent(name: String): Agent = Agent(this, name)

    /** Looking something up, under a stored router config or none. */
    public fun router(config: String = "", tags: Map<String, String> = emptyMap()): Router =
        Router(backend, config, tags)

    /**
     * Holds a conversation in writing: no call is joined, nothing is transcribed or spoken.
     *
     * The replies still come through the model with the same instructions, skills and knowledge
     * a call would have had, and stream into [AgentSession.conversation]. It returns with the
     * socket open, so the tools are already being answered.
     */
    public suspend fun chat(options: SessionOptions = SessionOptions()): AgentSession =
        follow(sessions.create(checked(options)), options.tools)

    /**
     * Puts an agent on a call and follows it.
     *
     * The agent joins as soon as this returns. Joining the same call from this device is what
     * the rtc module is for; this only starts the agent and gives you the state.
     */
    public suspend fun voice(callId: String, options: SessionOptions = SessionOptions()): AgentSession =
        follow(sessions.create(checked(options), callId), options.tools)

    /**
     * Follows a session this caller already has open, without creating one.
     *
     * Use this after a relaunch, on another screen, or for a fork. A session opened by somebody
     * else is not found, because reading one is reading a conversation.
     */
    public suspend fun attach(sessionId: String, tools: List<AgentTool> = emptyList()): AgentSession =
        follow(sessions.get(sessionId), tools)

    /**
     * Gets or creates a guest, so somebody can talk to an agent before they sign up.
     *
     * Remembered in [store] where there is one, so the next launch is the same guest and not a
     * second one with an empty history. Pass it to [setUser] to act as them. Moving a guest's
     * conversations onto the account they turn out to be is server-side only: it is the backend
     * that just authenticated them that knows which guest they were.
     */
    public suspend fun guestUser(options: GuestOptions = GuestOptions(), store: GuestStore? = null): Guest =
        backend.guest(options, store)

    /** Forgets the remembered guest without minting another, which is what signing out is. */
    public fun forgetGuest(store: GuestStore) {
        store.clear()
        backend.clearUser()
    }

    override fun close() {
        backend.close()
    }

    private suspend fun follow(session: Session, tools: List<AgentTool>): AgentSession =
        AgentSession(backend, session, tools).also { it.start() }

    private fun checked(options: SessionOptions): SessionOptions {
        val duplicate = options.tools.groupBy { it.name }.entries.firstOrNull { it.value.size > 1 }
        if (duplicate != null) {
            throw AgentsException.Configuration("${duplicate.key} is declared twice")
        }
        return options
    }
}

/**
 * One agent, and the conversations held with it.
 *
 * The name is the one the agent's config was synced under, which the router resolves; a name
 * that matches nothing is refused rather than starting an agent with no config.
 */
public class Agent internal constructor(private val agents: VisionAgents, public val name: String) {
    /** This agent's conversations: opening one names it, and listing is narrowed to it. */
    public val sessions: Sessions = Sessions(agents.backend, name)

    /** Holds a conversation with this agent in writing. See [VisionAgents.chat]. */
    public suspend fun chat(options: SessionOptions = SessionOptions()): AgentSession =
        agents.chat(options.copy(agent = options.agent ?: name))

    /** Holds a conversation with this agent in writing, answering these tools here. */
    public suspend fun chat(vararg tools: AgentTool): AgentSession =
        chat(SessionOptions(tools = tools.toList()))

    /** Puts this agent on a call and follows it. See [VisionAgents.voice]. */
    public suspend fun voice(callId: String, options: SessionOptions = SessionOptions()): AgentSession =
        agents.voice(callId, options.copy(agent = options.agent ?: name))
}
