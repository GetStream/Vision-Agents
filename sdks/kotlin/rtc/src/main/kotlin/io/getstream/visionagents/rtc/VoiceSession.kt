package io.getstream.visionagents.rtc

import android.content.Context
import io.getstream.result.Result
import io.getstream.video.android.core.Call
import io.getstream.video.android.core.CameraDirection
import io.getstream.video.android.core.StreamVideo
import io.getstream.video.android.core.StreamVideoBuilder
import io.getstream.video.android.core.socket.common.token.TokenProvider
import io.getstream.video.android.model.User
import io.getstream.visionagents.core.AgentSession
import io.getstream.visionagents.core.AgentTool
import io.getstream.visionagents.core.SessionOptions
import io.getstream.visionagents.core.VisionAgents
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Credentials for joining the Stream call an agent is on.
 *
 * Minting these is server-side only, so they come from the app's own backend rather than from
 * this device. A backend holding the Go, Python or Node SDK asks the router for a call token
 * and hands down what is here.
 */
public data class CallCredentials(
    val apiKey: String,
    val token: String,
    val userId: String,
    val userName: String,
    /** The Stream call to join, which is not the id the router holds the session by. */
    val callId: String,
    val callType: String,
)

/**
 * Asks the app's backend for credentials to join the call a session is holding.
 *
 * A function rather than a value because a token expires an hour in, and a long call that was
 * handed one value would drop when it did.
 */
public fun interface CallCredentialsProvider {
    public suspend fun credentials(sessionId: String): CallCredentials
}

/** Joining the call failed for a reason Stream's Video SDK gave rather than threw. */
public class CallFailed(message: String) : Exception(message)

/**
 * A spoken conversation: the agent on a call, and this device on the same call.
 *
 * Three things happen, in this order, and the order matters:
 *
 * 1. The router starts a session, which is what puts the agent on the call.
 * 2. The app's backend mints a token for joining that call, which names the Stream call to
 *    join. That is not the id the router holds the session by.
 * 3. Stream's Video SDK joins it, and audio starts flowing.
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
    /** The client this session built, which is the only one it may remove. */
    private var built: StreamVideo? = null

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
     * set before joining, so no front-facing frame is ever published.
     *
     * [credentials] is asked for the token to join with, and asked again when it expires.
     */
    public suspend fun join(camera: Boolean = false, credentials: CallCredentialsProvider) {
        if (joined.value != null) return
        try {
            val joining = credentials.credentials(session.id)
            val call = client(joining, credentials).call(joining.callType, joining.callId)
            call.camera.setDirection(CameraDirection.Back)
            call.camera.setEnabled(camera)
            call.microphone.setEnabled(true)
            // Created rather than only joined: which of this and the agent's own join arrives
            // first is a race, and the agent creates it the same way.
            when (val result = call.join(create = true)) {
                is Result.Failure -> throw CallFailed(result.value.message)
                is Result.Success -> Unit
            }
            // Remote audio otherwise plays out of the earpiece, and an agent you talk to
            // hands-free wants the speaker.
            call.speaker.setSpeakerPhone(true)
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

    /** Turns this device's camera on or off. Re-enabling keeps the back lens. */
    public fun setCameraEnabled(enabled: Boolean) {
        val call = joined.value ?: return
        if (enabled && call.camera.direction.value != CameraDirection.Back) {
            call.camera.flip()
        }
        call.camera.setEnabled(enabled)
        cameraOn.value = enabled
    }

    /**
     * Leaves this device's call. Closes the router session only if this device started it; an
     * attached device navigating away leaves it running.
     */
    public suspend fun leave() {
        leaveCall(closeSession = createdLocally)
    }

    /** Hangs up: leaves the call and ends the agent session, whoever started it. */
    public suspend fun end() {
        leaveCall(closeSession = true)
    }

    private suspend fun leaveCall(closeSession: Boolean) {
        joined.value?.leave()
        joined.value = null
        cameraOn.value = false
        if (built != null) {
            StreamVideo.removeClient()
            built = null
        }
        if (closeSession) session.close()
    }

    /**
     * Stream's Video SDK keeps one client per process and refuses a second. One the host already
     * has for this user is used as it is; one for anybody else is the host's to remove first.
     */
    private fun client(joining: CallCredentials, credentials: CallCredentialsProvider): StreamVideo {
        StreamVideo.instanceOrNull()?.takeIf { it.user.id == joining.userId }?.let { return it }
        // The provider is what the SDK calls when the token expires, an hour in. Handing it one
        // that asks the backend again is what keeps a long call from dropping.
        val refresh = object : TokenProvider {
            override suspend fun loadToken(): String = credentials.credentials(session.id).token
        }
        return StreamVideoBuilder(
            context = context.applicationContext,
            apiKey = joining.apiKey,
            user = User(id = joining.userId, name = joining.userName),
            token = joining.token,
            tokenProvider = refresh,
        ).build().also { built = it }
    }

    public companion object {
        /**
         * Starts an agent on a new call and prepares to join it.
         *
         * The call id is generated here unless one is given, so the common case, somebody
         * tapping "talk to the agent", needs no id from anywhere.
         */
        public suspend fun start(
            context: Context,
            agents: VisionAgents,
            callId: String = UUID.randomUUID().toString(),
            options: SessionOptions = SessionOptions(),
        ): VoiceSession = VoiceSession(context, agents.voice(callId, options), createdLocally = true)

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
