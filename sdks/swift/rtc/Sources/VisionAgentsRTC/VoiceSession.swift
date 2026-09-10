import Foundation
import Observation
import StreamVideo
import VisionAgentsCore

/// A spoken conversation: the agent on a call, and this device on the same call.
///
/// Three things happen, in this order, and the order matters:
///
/// 1. The router starts a session, which is what puts the agent on the call.
/// 2. The router mints a token for joining that call, which names the Stream call to join.
///    That is not the id the router holds the session by.
/// 3. Stream's Video SDK joins it, and audio starts flowing.
///
/// The transcript comes over the session socket rather than out of the call, so what is said
/// is readable even before anybody is listening to it. That is `session`, which is the same
/// `AgentSession` a text conversation uses.
@MainActor
@Observable
public final class VoiceSession {
    /// The conversation: the transcript, and what the agent is doing.
    public let session: AgentSession

    /// The Stream call this device is on, once it has joined.
    public private(set) var call: Call?

    /// Whether this device's microphone is on.
    public private(set) var isMuted = false

    /// Whether this device's camera is on.
    public private(set) var isCameraEnabled = false

    /// Why joining failed, or nil. Set rather than thrown because joining happens in a
    /// `task`, where there is nobody to throw to.
    public private(set) var failure: (any Error)?

    private let agents: VisionAgents
    private var video: StreamVideo?
    /// True when this device called `start`, false when it attached to a session something
    /// else started. Leaving closes only a session this device owns.
    private let createdLocally: Bool

    /// Starts an agent on a new call and prepares to join it.
    ///
    /// The call id is generated here unless one is given, so that the common case -- a person
    /// tapping "talk to the agent" -- needs no id from anywhere.
    public static func start(
        agents: VisionAgents,
        agent: String? = nil,
        callID: String = UUID().uuidString,
        tools: [AgentTool] = []
    ) async throws -> VoiceSession {
        let session = try await agents.voice(callID: callID, agent: agent, tools: tools)
        return VoiceSession(agents: agents, session: session, createdLocally: true)
    }

    /// Joins a call an agent is already on, without creating a session.
    ///
    /// `sessionID` is the id the router holds the session by, which is also a `CallRecord`'s
    /// `id`. That is what `callToken` and the events socket address.
    public static func attach(
        agents: VisionAgents,
        sessionID: String,
        tools: [AgentTool] = []
    ) async throws -> VoiceSession {
        let session = try await agents.attach(sessionID: sessionID, tools: tools)
        return VoiceSession(agents: agents, session: session, createdLocally: false)
    }

    private init(agents: VisionAgents, session: AgentSession, createdLocally: Bool) {
        self.agents = agents
        self.session = session
        self.createdLocally = createdLocally
    }

    /// Joins the call from this device.
    ///
    /// The agent is already there: it joined when the session was created. This is the other
    /// half of the conversation arriving. The microphone is on; the camera is off unless
    /// `camera` is true.
    ///
    /// A camera starts on the back lens, since what an agent is being shown is whatever the
    /// caller is pointing at. Both are join settings rather than changed afterwards, so no
    /// front-facing frame is ever published.
    public func join(camera: Bool = false) async {
        guard call == nil else { return }
        do {
            let credentials = try await agents.callToken(sessionID: session.id)

            // The token provider is what the SDK calls when the token expires, which it does
            // an hour in. Handing it a closure that asks the router again is what keeps a long
            // call from dropping; handing it the same expired token, as the convenience
            // initialiser does by default, would not.
            let video = StreamVideo(
                apiKey: credentials.apiKey,
                user: User(id: credentials.userID, name: credentials.userName),
                token: UserToken(rawValue: credentials.token),
                tokenProvider: { [agents, session] result in
                    Task {
                        do {
                            let refreshed = try await agents.callToken(
                                sessionID: session.id, userID: credentials.userID)
                            result(.success(UserToken(rawValue: refreshed.token)))
                        } catch {
                            result(.failure(error))
                        }
                    }
                })
            self.video = video

            let call = video.call(callType: credentials.callType, callId: credentials.callID)
            // Created rather than only joined: which of the two arrives first is a race, and
            // the agent's own join creates it the same way.
            try await call.join(
                create: true,
                callSettings: CallSettings(videoOn: camera, cameraPosition: .back))
            if camera {
                try await call.camera.enable()
            } else {
                try await call.camera.disable()
            }
            isCameraEnabled = camera
            try await call.microphone.enable()
            // Remote audio plays through the audio session on its own once joined, but out of
            // the earpiece. An agent you talk to hands-free wants the speaker.
            try await call.speaker.enableSpeakerPhone()
            self.call = call
        } catch is CancellationError {
            return
        } catch {
            failure = error
        }
    }

    /// Turns this device's microphone on or off. The agent stays on the call either way.
    public func setMuted(_ muted: Bool) async {
        guard let call else { return }
        do {
            try await muted ? call.microphone.disable() : call.microphone.enable()
            isMuted = muted
        } catch {
            failure = error
        }
    }

    /// Turns this device's camera on or off. Re-enabling keeps the back lens.
    public func setCameraEnabled(_ enabled: Bool) async {
        guard let call else { return }
        do {
            if enabled {
                if call.camera.direction != .back {
                    try await call.camera.flip()
                }
                try await call.camera.enable()
            } else {
                try await call.camera.disable()
            }
            isCameraEnabled = enabled
        } catch {
            failure = error
        }
    }

    /// Leaves this device's RTC call. Closes the router session only if this
    /// device started it; an attached device navigating away leaves it running.
    public func leave() async {
        await leaveCall(closeSession: createdLocally)
    }

    /// Hangs up: leaves the RTC call and ends the agent session, whoever started it.
    public func end() async {
        await leaveCall(closeSession: true)
    }

    private func leaveCall(closeSession: Bool) async {
        call?.leave()
        call = nil
        video = nil
        isCameraEnabled = false
        if closeSession {
            await session.close()
        }
    }
}
