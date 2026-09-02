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

    /// Why joining failed, or nil. Set rather than thrown because joining happens in a
    /// `task`, where there is nobody to throw to.
    public private(set) var failure: (any Error)?

    private let agents: VisionAgents
    private var video: StreamVideo?

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
        return VoiceSession(agents: agents, session: session)
    }

    private init(agents: VisionAgents, session: AgentSession) {
        self.agents = agents
        self.session = session
    }

    /// Joins the call from this device, with the microphone on and the camera off.
    ///
    /// The agent is already there: it joined when the session was created. This is the other
    /// half of the conversation arriving.
    public func join() async {
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
            try await call.join(create: true)
            try await call.camera.disable()
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

    /// Leaves the call and ends the session, so the agent leaves too.
    public func leave() async {
        call?.leave()
        call = nil
        video = nil
        await session.close()
    }
}
