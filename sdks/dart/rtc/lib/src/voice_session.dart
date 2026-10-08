import 'dart:math';

import 'package:stream_video_flutter/stream_video_flutter.dart' as video;
import 'package:vision_agents_core/vision_agents_core.dart';

/// Credentials for joining the Stream call an agent is on.
///
/// Minting these is server-side only, so they come from the app's own backend rather than
/// from this device. A backend holding the Go or Python SDK asks the router for a call token
/// and hands down what is here.
final class CallCredentials {
  const CallCredentials({
    required this.apiKey,
    required this.token,
    required this.userId,
    required this.callId,
    this.userName = '',
    this.callType = 'default',
  });

  final String apiKey;
  final String token;
  final String userId;
  final String userName;

  /// The Stream call to join, which is not the id the router holds the session by.
  final String callId;
  final String callType;
}

/// Asks the app's backend for credentials to join the call a session is holding.
///
/// A function rather than a value because a token expires an hour in, and a long call that
/// was handed one value would drop when it did.
typedef CallCredentialsProvider = Future<CallCredentials> Function(String sessionId);

/// Where a spoken conversation is: joined or not, microphone, camera, and why it failed.
final class VoiceState {
  const VoiceState({this.call, this.isMuted = false, this.isCameraEnabled = false, this.failure});

  /// The Stream call this device is on, once it has joined.
  final video.Call? call;
  final bool isMuted;
  final bool isCameraEnabled;

  /// Why joining or a change failed, or null. Kept rather than thrown, since joining is
  /// usually started from a widget where there is nobody to throw to.
  final Object? failure;

  bool get isJoined => call != null;

  VoiceState _with({video.Call? call, bool? isMuted, bool? isCameraEnabled, Object? failure}) =>
      VoiceState(
        call: call ?? this.call,
        isMuted: isMuted ?? this.isMuted,
        isCameraEnabled: isCameraEnabled ?? this.isCameraEnabled,
        failure: failure ?? this.failure,
      );
}

/// Joining or changing the call failed; [message] is what Stream Video said.
final class CallFailure implements Exception {
  const CallFailure(this.message);

  final String message;

  @override
  String toString() => message;
}

/// A spoken conversation: the agent on a call, and this device on the same call.
///
/// Three things happen, in this order, and the order matters:
///
/// 1. The router starts a session, which is what puts the agent on the call.
/// 2. The app's backend mints a token for joining that call, which names the Stream call to
///    join. That is not the id the router holds the session by.
/// 3. Stream Video joins it, and audio starts flowing.
///
/// The transcript comes over the session socket rather than out of the call, so what is said
/// is readable even before anybody is listening to it. That is [session], the same
/// [AgentSession] a text conversation uses.
final class VoiceSession {
  VoiceSession._(this.session, {required bool createdLocally}) : _createdLocally = createdLocally;

  /// Starts an agent on a new call and prepares to join it. The call id is generated unless
  /// one is given, so a person tapping "talk to the agent" needs no id from anywhere.
  static Future<VoiceSession> start(
    VisionAgents agents, {
    String? agent,
    String? callId,
    SessionOptions options = const SessionOptions(),
  }) async {
    final id = callId ?? _callId();
    final session = await (agent == null
        ? agents.voice(id, options)
        : agents.agent(agent).voice(id, options));
    return VoiceSession._(session, createdLocally: true);
  }

  /// Joins a call an agent is already on, without creating a session.
  ///
  /// [sessionId] is the id the router holds the session by. A session this user did not open
  /// is not found, since reading one is reading a conversation.
  static Future<VoiceSession> attach(
    VisionAgents agents,
    String sessionId, {
    List<AgentTool> tools = const [],
  }) async => VoiceSession._(await agents.attach(sessionId, tools: tools), createdLocally: false);

  /// The conversation: the transcript, and what the agent is doing.
  final AgentSession session;

  /// True when this device called [start]. Leaving closes only a session this device owns.
  final bool _createdLocally;
  final _state = LiveValueController(const VoiceState());
  video.StreamVideo? _video;
  bool _joining = false;

  LiveValue<VoiceState> get state => _state;
  video.Call? get call => _state.value.call;

  /// Joins the call from this device.
  ///
  /// The agent is already there: it joined when the session was created. The microphone is
  /// on and the speaker is the loudspeaker; the camera is off unless [camera] is true, and
  /// starts on the back lens, since what an agent is shown is whatever the caller points at.
  /// Both are join settings rather than changed afterwards, so no front-facing frame is ever
  /// published.
  ///
  /// [credentials] is asked for the token to join with, and asked again when it expires.
  Future<void> join({bool camera = false, required CallCredentialsProvider credentials}) async {
    if (_joining || call != null) {
      return;
    }
    _joining = true;
    try {
      final joining = await credentials(session.id);
      // `create` rather than the singleton constructor: a library installing Stream Video's
      // singleton would collide with a host that has its own.
      final client = video.StreamVideo.create(
        joining.apiKey,
        user: video.User.regular(userId: joining.userId, name: joining.userName),
        userToken: joining.token,
        // Asked when the token expires, which it does an hour in. Asking the backend again is
        // what keeps a long call from dropping.
        tokenLoader: (_) async => (await credentials(session.id)).token,
      );
      _video = client;
      final joined = client.makeCall(
        callType: video.StreamCallType.fromString(joining.callType),
        id: joining.callId,
      );
      // `join` creates the call too: which of the agent and this device arrives first is a
      // race.
      final result = await joined.join(
        connectOptions: video.CallConnectOptions(
          microphone: video.TrackOption.enabled(),
          camera: video.TrackOption.fromSetting(enabled: camera),
          cameraFacingMode: video.FacingMode.environment,
          // Remote audio otherwise plays out of the earpiece, and an agent you talk to
          // hands-free wants the speaker.
          speakerDefaultOn: true,
        ),
      );
      if (result case video.Failure(:final error)) {
        await client.dispose();
        _video = null;
        _state.value = _state.value._with(failure: CallFailure(error.message));
        return;
      }
      _state.value = _state.value._with(call: joined, isMuted: false, isCameraEnabled: camera);
    } on Exception catch (error) {
      _state.value = _state.value._with(failure: error);
    } finally {
      _joining = false;
    }
  }

  /// Turns this device's microphone on or off. The agent stays on the call either way.
  Future<void> setMuted(bool muted) async {
    final call = this.call;
    if (call == null) {
      return;
    }
    if (_failed(await call.setMicrophoneEnabled(enabled: !muted))) {
      return;
    }
    _state.value = _state.value._with(isMuted: muted);
  }

  /// Turns this device's camera on or off. Turning it back on keeps the back lens.
  Future<void> setCameraEnabled(bool enabled) async {
    final call = this.call;
    if (call == null) {
      return;
    }
    final result = await call.setCameraEnabled(
      enabled: enabled,
      constraints: const video.CameraConstraints(facingMode: video.FacingMode.environment),
    );
    if (_failed(result)) {
      return;
    }
    _state.value = _state.value._with(isCameraEnabled: enabled);
  }

  /// Leaves this device's call. Closes the router session only if this device started it;
  /// an attached device navigating away leaves it running.
  Future<void> leave() => _leave(closeSession: _createdLocally);

  /// Hangs up: leaves the call and ends the agent session, whoever started it.
  Future<void> end() => _leave(closeSession: true);

  Future<void> _leave({required bool closeSession}) async {
    final call = this.call;
    final client = _video;
    _video = null;
    _state.value = VoiceState(failure: _state.value.failure);
    if (call != null) {
      await call.leave();
    }
    await client?.dispose();
    if (closeSession) {
      await session.close();
    }
  }

  bool _failed(video.Result<video.None> result) {
    if (result case video.Failure(:final error)) {
      _state.value = _state.value._with(failure: CallFailure(error.message));
      return true;
    }
    return false;
  }

  static String _callId() {
    final random = Random.secure();
    return List.generate(16, (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0')).join();
  }
}
