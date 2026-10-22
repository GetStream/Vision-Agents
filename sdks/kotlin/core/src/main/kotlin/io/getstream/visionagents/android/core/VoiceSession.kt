package io.getstream.visionagents.android.core

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import io.getstream.result.Result
import io.getstream.video.android.core.Call
import io.getstream.video.android.core.CameraDirection
import io.getstream.video.android.core.StreamVideo
import io.getstream.video.android.core.StreamVideoBuilder
import io.getstream.video.android.core.notifications.internal.service.DefaultCallConfigurations
import io.getstream.video.android.core.permission.android.StreamPermissionCheck
import io.getstream.video.android.core.socket.common.token.TokenProvider
import io.getstream.video.android.model.User
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

private const val KIND = "video"

/** The last client built here, so the process-wide instance can be told apart from the app's. */
@Volatile private var builtHere: StreamVideo? = null

/**
 * Hands over the app's own Stream Video client, so every call is joined on it.
 *
 * Stream Video keeps one client per process, so an app with one of its own hands it over rather
 * than have another built. It is never removed here.
 */
public fun VisionAgents.use(video: StreamVideo) {
    backend.give(KIND, video)
}

/**
 * A spoken conversation: the agent on a call, and this device on the same call.
 *
 * Two things happen, in this order:
 *
 * 1. The router starts a session with voice on, which puts the agent on the session's own
 *    call, `agent:<session id>`.
 * 2. Stream's Video SDK joins that call as the user `setUser` named, and audio starts flowing.
 *
 * The transcript comes over the session socket rather than out of the call, so what is said is
 * readable even before anybody is listening to it. That is [session], the same [AgentSession] a
 * text conversation uses. The app must hold `RECORD_AUDIO`, and `CAMERA` for a camera, before
 * [join]: asking for them is a screen's job, not this.
 */
public class VoiceSession private constructor(
    private val context: Context,
    /** The conversation: the transcript, and what the agent is doing. */
    public val session: AgentSession,
    /** True when this device started the session; leaving closes only one this device owns. */
    private val createdLocally: Boolean,
) {
    private val joined = MutableStateFlow<Call?>(null)
    private val muted = MutableStateFlow(false)
    private val cameraOn = MutableStateFlow(false)
    private val stoppedBy = MutableStateFlow<Throwable?>(null)

    /** What the camera showed lately, while it is on, for the agent to look at. */
    private val frames = CameraFrames()
    private var looking: CameraFilter? = null

    /** The Stream call this device is on, once it has joined. */
    public val call: StateFlow<Call?> = joined.asStateFlow()

    /** Whether this device's microphone is off. */
    public val isMuted: StateFlow<Boolean> = muted.asStateFlow()

    /** Whether this device's camera is on. */
    public val isCameraEnabled: StateFlow<Boolean> = cameraOn.asStateFlow()

    /** Why joining failed, or null. Set rather than thrown, since joining runs in an effect. */
    public val failure: StateFlow<Throwable?> = stoppedBy.asStateFlow()

    /**
     * Joins the call from this device.
     *
     * The agent is already there: it joined when the session was created. The microphone is on
     * and plays through the speaker; the camera is off unless [camera] is true, and starts on
     * the back lens, since what an agent is shown is whatever the caller is pointing at. Both are
     * set before joining, so no front-facing frame is ever published. While the camera is on, the
     * agent can look through it: see [setCameraEnabled].
     *
     * It joins on the client handed over with [use], or else on one built from the agents' key,
     * user and token, which every voice session then shares.
     */
    public suspend fun join(camera: Boolean = false) {
        if (joined.value != null) return
        try {
            val video = client()
            val type = session.session.callType
            // An agent's call is a voice call that may add the back camera, so its foreground
            // service is Stream's audio-call one, which asks only for the microphone. The default
            // refuses to start without the camera permission, leaving the call with no service
            // holding it. A type the app configured itself is left as it is.
            val services = video.state.callConfigRegistry
            if (services.get(type) == DefaultCallConfigurations.default) {
                services.register(type, DefaultCallConfigurations.audioCall)
            }
            val call = video.call(type, session.session.callId)
            call.camera.setDirection(CameraDirection.Back)
            call.camera.setEnabled(camera)
            call.microphone.setEnabled(true)
            // Created rather than only joined: which of this and the agent's own join arrives
            // first is a race, and the agent creates it the same way.
            when (val result = call.join(create = true)) {
                is Result.Failure -> throw AgentsException.CallFailed(result.value.message)
                is Result.Success -> Unit
            }
            // Remote audio otherwise plays out of the earpiece, and an agent you talk to
            // hands-free wants the speaker.
            call.speaker.setSpeakerPhone(true)
            if (camera) look(call)
            cameraOn.value = camera
            muted.value = false
            joined.value = call
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // Stream's SDK and the host's provider throw whatever they throw; all of it is a
            // call that did not start, which is what `failure` reports.
            stoppedBy.value = e
        }
    }

    /** Turns this device's microphone off or on. The agent stays on the call either way. */
    public fun setMuted(muted: Boolean) {
        val call = joined.value ?: return
        call.microphone.setEnabled(!muted)
        this.muted.value = muted
    }

    /**
     * Turns this device's camera on or off. Re-enabling keeps the back lens.
     *
     * While it is on, the agent can look through it: its vision skill asks for what the camera
     * showed when a question was put, and [session] answers with those frames, sampled twice a
     * second from what the call publishes. Turned off, what it showed is forgotten.
     */
    public fun setCameraEnabled(enabled: Boolean) {
        val call = joined.value ?: return
        if (enabled && call.camera.direction.value != CameraDirection.Back) {
            call.camera.flip()
        }
        call.camera.setEnabled(enabled)
        if (enabled) look(call) else stopLooking()
        cameraOn.value = enabled
    }

    /**
     * Leaves this device's call. Closes the router session only if this device started it; an
     * attached device navigating away leaves it running. The Video client stays connected for
     * the next call, until [VisionAgents.disconnect].
     */
    public suspend fun leave() {
        leaveCall(closeSession = createdLocally)
    }

    /** Hangs up: leaves the call and ends the agent session, whoever started it. */
    public suspend fun end() {
        leaveCall(closeSession = true)
    }

    /** Samples the call's camera for the agent, and lets [session] answer with what it showed. */
    private fun look(call: Call) {
        if (looking != null) return
        looking = CameraFilter(call::videoFilter, frames).also { it.start() }
        session.cameraFrames = frames
    }

    private fun stopLooking() {
        looking?.stop()
        looking = null
        frames.clear()
        session.cameraFrames = null
    }

    private suspend fun leaveCall(closeSession: Boolean) {
        stopLooking()
        joined.value?.leave()
        joined.value = null
        cameraOn.value = false
        if (closeSession) session.close()
    }

    /**
     * The shared Video client for the agents' user.
     *
     * Stream's Video SDK keeps one client per process and refuses a second, so one the host built
     * without handing it over is used if it is signed in as the same user and refused otherwise.
     */
    private suspend fun client(): StreamVideo {
        val backend = session.backend
        val existing = StreamVideo.instanceOrNull()
        if (existing != null && existing !== builtHere) {
            if (existing.user.id != backend.user?.id) {
                throw AgentsException.Configuration(
                    "this process already has a StreamVideo, signed in as ${existing.user.id}: " +
                        "sign it in as the user setUser named and hand it over with " +
                        "agents.use(video), since Stream Video refuses a second",
                )
            }
            backend.give(KIND, existing)
        }
        val video = backend.shared(
            KIND,
            open = { build(backend, it) },
            disconnect = { StreamVideo.removeClient() },
        )
        val user = backend.user
        if (user != null && video.user.id != user.id) {
            throw AgentsException.Configuration(
                "the video client is connected as ${video.user.id} but setUser named ${user.id}: " +
                    "connect it as the same user, or the call is somebody else's",
            )
        }
        return video
    }

    private fun build(backend: Backend, credentials: StreamCredentials): StreamVideo {
        // The provider is what the SDK calls when the token expires, an hour in. Asking the
        // agents' token provider again is what keeps a long call from dropping.
        val refresh = object : TokenProvider {
            override suspend fun loadToken(): String = backend.streamCredentials(refresh = true).token
        }
        return StreamVideoBuilder(
            context = context.applicationContext,
            apiKey = credentials.apiKey,
            user = User(id = credentials.user.id, name = credentials.user.name),
            token = credentials.token,
            tokenProvider = refresh,
            permissionCheck = VoicePermissions,
        ).build().also { builtHere = it }
    }

    public companion object {
        /**
         * Starts an agent on the new session's own call, `agent:<session id>`, and prepares to
         * join it.
         */
        public suspend fun start(
            context: Context,
            agents: VisionAgents,
            options: SessionOptions = SessionOptions(),
        ): VoiceSession = VoiceSession(context, agents.voice(options), createdLocally = true)

        /**
         * Joins a call an agent is already on, without creating a session.
         *
         * [sessionId] is the id the router holds the session by. A session this caller did not
         * open is not found, since reading one is reading a conversation.
         */
        public suspend fun attach(
            context: Context,
            agents: VisionAgents,
            sessionId: String,
            tools: List<AgentTool> = emptyList(),
        ): VoiceSession = VoiceSession(context, agents.attach(sessionId, tools), createdLocally = false)
    }
}

/**
 * What a call with an agent needs from Android: the microphone, and the camera only while it is
 * on. Stream's own check goes by what the call allows, and an agent's call allows video, so it
 * would refuse a voice call without the camera permission.
 */
private object VoicePermissions : StreamPermissionCheck {
    override fun checkAndroidPermissionsGroup(context: Context, call: Call): Pair<Boolean, Set<String>> {
        val needed = buildSet {
            add(Manifest.permission.RECORD_AUDIO)
            if (call.camera.isEnabled.value) add(Manifest.permission.CAMERA)
        }
        val missing = needed.filterTo(mutableSetOf()) {
            context.checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        return missing.isEmpty() to missing
    }

    @Deprecated("Use checkAndroidPermissionsGroup instead")
    override fun checkAndroidPermissions(context: Context, call: Call): Boolean =
        checkAndroidPermissionsGroup(context, call).first
}
