import 'package:stream_video_flutter/stream_video_flutter.dart' as video;
import 'package:vision_agents_core/vision_agents_core.dart';

import 'camera_jpeg.dart';
import 'camera_shots.dart';

const _kind = 'video';

/// The app's own Stream Video client.
extension VideoClients on VisionAgents {
  /// Hands over the app's own Stream Video client, so every call is joined on it rather than
  /// on a second one. It must be connected as the agents' `userId`, and [disconnect] leaves it
  /// alone.
  void useVideo(video.StreamVideo client) {
    if (client.currentUser.id != backend.userId) {
      throw ArgumentError(
        'the video client is for ${client.currentUser.id} but these agents are for '
        '${backend.userId}: hand over a client for the same user',
      );
    }
    backend.give(_kind, client);
  }
}

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
/// Two things happen, in this order, and the order matters:
///
/// 1. The router starts a session, which puts the agent on the session's own call,
///    `agent:<session id>`.
/// 2. Stream Video joins it with the agents' key, user and token, and audio starts flowing.
///
/// The transcript comes over the session socket rather than out of the call, so what is said
/// is readable even before anybody is listening to it. That is [session], the same
/// [AgentSession] a text conversation uses.
final class VoiceSession {
  VoiceSession._(this.session, {required bool createdLocally}) : _createdLocally = createdLocally;

  /// Starts an agent on a new session's call and prepares to join it, so a person tapping
  /// "talk to the agent" needs no id from anywhere.
  static Future<VoiceSession> start(
    VisionAgents agents, {
    String? agent,
    SessionOptions options = const SessionOptions(),
  }) async {
    final session = await (agent == null
        ? agents.voice(options)
        : agents.agent(agent).voice(options));
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
  bool _joining = false;
  late final _shots = CameraShots(_shot, cameraJpeg);

  /// What [session] takes pictures with while the camera is on, kept so it can be told apart
  /// from one the app set.
  late final _camera = _shots.look;

  LiveValue<VoiceState> get state => _state;
  video.Call? get call => _state.value.call;

  /// Joins the call from this device.
  ///
  /// The agent is already there: it joined when the session was created. The microphone is
  /// on and the speaker is the loudspeaker; the camera is off unless [camera] is true, and
  /// starts on the back lens, since what an agent is shown is whatever the caller points at.
  /// Both are join settings rather than changed afterwards, so no front-facing frame is ever
  /// published. While the camera is on, the agent can look through it, as [setCameraEnabled]
  /// says.
  ///
  /// It joins on the client handed over with `useVideo`, or else on one built from the
  /// agents' key, user and token, which every voice session then shares. Without a key, user
  /// and token there is nothing to join with, which is an [ArgumentError].
  Future<void> join({bool camera = false}) async {
    if (_joining || call != null) {
      return;
    }
    _joining = true;
    try {
      final joined = (await _client()).makeCall(
        callType: video.StreamCallType.fromString(session.session.callType),
        id: session.session.callId,
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
        _state.value = _state.value._with(failure: CallFailure(error.message));
        return;
      }
      _state.value = _state.value._with(call: joined, isMuted: false, isCameraEnabled: camera);
      if (camera) {
        _look(joined);
      }
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
  ///
  /// While it is on, the agent can look through it: its vision skill asks for what the camera
  /// showed when a question was put, and [session] answers with those frames. Turned off,
  /// what it showed is forgotten.
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
    if (enabled) {
      _look(call);
    } else {
      _stopLooking();
    }
  }

  /// Leaves this device's call. Closes the router session only if this device started it;
  /// an attached device navigating away leaves it running. The Video client stays connected
  /// for the next call, until [VisionAgents.disconnect].
  Future<void> leave() => _leave(closeSession: _createdLocally);

  /// Hangs up: leaves the call and ends the agent session, whoever started it.
  Future<void> end() => _leave(closeSession: true);

  Future<void> _leave({required bool closeSession}) async {
    _stopLooking();
    final call = this.call;
    _state.value = VoiceState(failure: _state.value.failure);
    if (call != null) {
      await call.leave();
    }
    if (closeSession) {
      await session.close();
    }
  }

  Future<video.StreamVideo> _client() {
    final backend = session.backend;
    return backend.shared<video.StreamVideo>(
      _kind,
      // `create` rather than the singleton constructor: a library installing Stream Video's
      // singleton would collide with a host that has its own.
      open: (credentials) async => video.StreamVideo.create(
        credentials.apiKey,
        user: video.User.regular(userId: credentials.userId),
        userToken: credentials.token,
        // Asked when the token expires, which it does an hour in. Asking the agents' token
        // provider again is what keeps a long call from dropping.
        tokenLoader: (_) async => (await backend.streamCredentials()).token,
      ),
      disconnect: (client) => client.dispose(),
    );
  }

  /// Lets [session] answer the agent with pictures from the call's camera.
  ///
  /// A call left while its camera was being turned on is not looked through.
  void _look(video.Call call) {
    if (!identical(this.call, call)) {
      return;
    }
    session.camera = _camera;
  }

  void _stopLooking() {
    _shots.stop();
    if (identical(session.camera, _camera)) {
      session.camera = null;
    }
  }

  /// The picture to take with the call's camera now, or null when it is not sending video.
  CameraShot? _shot() {
    // Only while the call is sending video: a camera turned off some other way, by Stream
    // Video's own controls or the app's, has no frame to give.
    final call = this.call;
    final local = call?.state.value.localParticipant;
    if (call == null || local == null || !local.isVideoEnabled) {
      return null;
    }
    final track = call.getTrack(local.trackIdPrefix, video.SfuTrackType.video);
    if (track == null) {
      return null;
    }
    return (
      track: track.mediaTrack.id,
      take: () async => (await track.captureScreenshot())?.asUint8List(),
    );
  }

  bool _failed(video.Result<video.None> result) {
    if (result case video.Failure(:final error)) {
      _state.value = _state.value._with(failure: CallFailure(error.message));
      return true;
    }
    return false;
  }
}
