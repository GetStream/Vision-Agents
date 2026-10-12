import Foundation
import Observation
import StreamVideo

private let kind = "video"

extension VisionAgents {
    /// Hands over the app's own Stream Video client, so every call is joined on it.
    ///
    /// An app with a `StreamVideo` of its own must hand it over: building another makes that
    /// one the process's current instance, which Stream's call views and CallKit read. It is
    /// never disconnected here.
    public func use(_ video: StreamVideo) {
        backend.give(kind, video)
    }
}

/// A spoken conversation: the agent on a call, and this device on the same call.
///
/// Two things happen, in this order:
///
/// 1. The router starts a session with voice on, which puts the agent on the session's own
///    call, `agent:<session id>`.
/// 2. Stream's Video SDK joins that call as the user `setUser` named, and audio starts
///    flowing.
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
    /// True when this device called `start`, false when it attached to a session something
    /// else started. Leaving closes only a session this device owns.
    private let createdLocally: Bool

    /// Starts an agent on the new session's own call, `agent:<session id>`, and prepares to
    /// join it.
    public static func start(
        agents: VisionAgents,
        agent: String? = nil,
        tools: [AgentTool] = []
    ) async throws -> VoiceSession {
        let session = try await agents.voice(agent: agent, tools: tools)
        return VoiceSession(agents: agents, session: session, createdLocally: true)
    }

    /// Joins a call an agent is already on, without creating a session.
    ///
    /// `sessionID` is the id the router holds the session by, which is what the events socket
    /// addresses. A session this device did not open is not found, since reading one is
    /// reading a conversation.
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
    ///
    /// It joins on the `StreamVideo` handed over with `use`, or else on one built for the
    /// agents' key and user, which every voice session then shares.
    public func join(camera: Bool = false) async {
        guard call == nil else { return }
        do {
            let backend = agents.backend
            let video = try await backend.shared(
                kind, open: { build(backend, $0) }, disconnect: { await $0.disconnect() })
            if let user = backend.user, video.user.id != user.id {
                throw AgentsError.configuration(
                    "the video client is connected as \(video.user.id) but setUser named "
                        + "\(user.id): connect it as the same user, or the call is somebody else's")
            }

            let call = video.call(callType: session.session.callType, callId: session.session.callID)
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
        isCameraEnabled = false
        if closeSession {
            await session.close()
        }
    }
}

/// A client for the agents' user. The token provider is what the SDK calls when the token
/// expires, an hour in, so asking the agents again is what keeps a long call from dropping.
private func build(_ backend: Backend, _ credentials: StreamCredentials) -> StreamVideo {
    let user = credentials.user
    return StreamVideo(
        apiKey: credentials.apiKey,
        user: .init(
            id: user.id, name: user.name.isEmpty ? nil : user.name, imageURL: URL(string: user.image)),
        token: UserToken(rawValue: credentials.token),
        tokenProvider: { result in
            Task {
                do {
                    result(.success(UserToken(rawValue: try await backend.streamCredentials(refresh: true).token)))
                } catch {
                    result(.failure(error))
                }
            }
        })
}
