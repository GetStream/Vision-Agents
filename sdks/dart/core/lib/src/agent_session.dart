import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'agent_event.dart';
import 'backend.dart';
import 'command.dart';
import 'conversation.dart';
import 'errors.dart';
import 'live_value.dart';
import 'models.dart';
import 'sessions.dart';
import 'socket.dart';
import 'tools.dart';

/// How many events are held for one caller of [AgentSession.events] who is not reading them.
///
/// Past this that caller's stream is finished rather than the socket stalled or events
/// dropped without a word. Nothing is lost by it: the transcript and tool calls are handled
/// by the session, not by whoever is reading events.
const _bufferedEvents = 256;

/// Whether the socket is carrying the conversation.
sealed class Connection {
  const Connection();

  static const notStarted = NotStarted._();
  static const connected = Connected._();
}

/// [AgentSession.start] has not been called.
final class NotStarted extends Connection {
  const NotStarted._();
}

/// The socket is open.
final class Connected extends Connection {
  const Connected._();
}

/// The socket has stopped, for good.
final class Disconnected extends Connection {
  const Disconnected([this.failure]);

  /// Why, or null for a conversation that ended normally or was closed here.
  final AgentsException? failure;

  @override
  bool operator ==(Object other) => other is Disconnected && other.failure == failure;

  @override
  int get hashCode => failure.hashCode;
}

/// A live conversation, as state a UI can build from.
///
/// The state is two [LiveValue]s, [conversation] and [connection], so a widget rebuilds from
/// exactly what it shows. They change in one place, the read loop below, and nowhere else.
/// Tool calls are answered in that loop too, so answering one never depends on anybody
/// listening to anything.
final class AgentSession {
  AgentSession(
    Session session, {
    required Backend backend,
    required Sessions sessions,
    List<AgentTool> tools = const [],
    this.interim = false,
  }) : _session = session,
       _backend = backend,
       _sessions = sessions,
       _tools = {for (final tool in tools.reversed) tool.name: tool};

  /// Where this conversation is held, and who is holding it.
  Backend get backend => _backend;

  /// The session the router opened, as of the last [update].
  Session get session => _session;
  Session _session;

  /// Asking the agent something, this conversation's turns as the router wrote them down,
  /// and rewinding to one.
  ///
  /// What is asked here shows in [conversation] at once, before the router has answered.
  late final Responses responses = heldResponses(_sessions, session.id, asking: _asked);

  /// Also report what somebody is part way through saying, as `hearing` events.
  final bool interim;

  /// Takes pictures with this device's camera when the agent asks to see, or null to leave
  /// the asking to somebody else.
  ///
  /// The hook `vision_agents_rtc` answers the agent's vision skill through while the call's
  /// camera is on. It is called with the moment the question was taken and how many pictures
  /// are wanted, 1 to 8, and returns JPEGs with when each was taken. None at all tells the
  /// agent the camera has shown nothing recent.
  Future<List<({Uint8List jpeg, DateTime capturedAt})>> Function(DateTime at, int limit)? camera;

  final Backend _backend;
  final Sessions _sessions;
  final Map<String, AgentTool> _tools;
  final _conversation = LiveValueController(const Conversation());
  final _connection = LiveValueController<Connection>(Connection.notStarted);
  final _approvals = LiveValueController<List<ToolApprovalRequest>>(const []);
  final Set<_Subscriber> _subscribers = {};
  final Set<String> _running = {};
  final Set<String> _cancelled = {};
  int _pictures = 0;
  SessionSocket? _socket;
  StreamSubscription<AgentEvent>? _pump;

  /// What the router holds this session by, which addresses it and its socket.
  String get id => session.id;

  /// The transcript and what the agent is doing.
  LiveValue<Conversation> get conversation => _conversation;

  /// Whether the socket is carrying the conversation, and why it stopped.
  LiveValue<Connection> get connection => _connection;

  /// Calls of tools that ask first, waiting for the person to answer with [decide], oldest
  /// first.
  LiveValue<List<ToolApprovalRequest>> get approvals => _approvals;

  List<Turn> get turns => _conversation.value.turns;
  ConversationState get state => _conversation.value.state;
  bool get isConnected => _connection.value is Connected;

  /// Why the socket stopped, or null. A conversation that ended normally has none.
  AgentsException? get failure => switch (_connection.value) {
    Disconnected(:final failure) => failure,
    _ => null,
  };

  /// Opens the socket and starts following the conversation. Calling it again does nothing.
  ///
  /// Throws, and reports it on [connection], when the socket cannot be opened.
  Future<void> start() async {
    if (_socket != null) {
      return;
    }
    final socket = SessionSocket(
      await _backend.socketUri('/v1/agents/sessions/${Uri.encodeComponent(id)}/events', {
        // Decisions arrive several times a second and are for somebody watching a call, not
        // for an app holding one.
        'decisions': 'false',
        if (interim) 'interim': 'true',
      }),
    );
    _socket = socket;

    final Stream<AgentEvent> events;
    try {
      events = await socket.open();
    } on AgentsException catch (error) {
      _stopped(error);
      rethrow;
    }
    _connection.value = Connection.connected;
    _pump = events.listen(
      _apply,
      onError: (Object error) =>
          _stopped(error is AgentsException ? error : TransportException(error)),
      onDone: () => _stopped(null),
    );
  }

  /// Everything the conversation did from now on, until it ends.
  ///
  /// Each call is a stream of its own, so two listeners each see every event. A listener
  /// more than 256 events behind is finished rather than buffered without bound.
  Stream<AgentEvent> events() {
    final subscriber = _Subscriber();
    if (_connection.value is Disconnected) {
      subscriber.controller.close();
      return subscriber.controller.stream;
    }
    _subscribers.add(subscriber);
    subscriber.controller.onCancel = () => _subscribers.remove(subscriber);
    return subscriber.controller.stream.map((event) {
      subscriber.pending--;
      return event;
    });
  }

  /// Speaks this without going through the model.
  void say(String text) => _held().send(SayCommand(text));

  /// Abandons the reply in flight.
  void interrupt() => _held().send(const InterruptCommand());

  /// Answers a call waiting in [approvals].
  ///
  /// Allowed, the tool runs and its result goes back as it would have. Declined, it never
  /// runs: the model is told the person declined, and the call's step shows [summary]. A call
  /// that is not waiting is left alone, so answering twice does nothing. If the answer cannot
  /// be sent, the call goes on waiting.
  void decide(String toolCallId, {required bool allowed, String summary = ''}) {
    final waiting = _approvals.value;
    final index = waiting.indexWhere((request) => request.id == toolCallId);
    if (index < 0) {
      return;
    }
    final request = waiting[index];
    final call = request.call;
    _held().send(
      ToolApprovalCommand(
        call.id,
        allowed: allowed,
        summary: summary,
        requestId: call.requestId,
        turnId: call.turnId,
      ),
    );
    _approvals.value = [...waiting]..removeAt(index);
    if (allowed) {
      _run(call, request._tool);
    } else {
      _held().send(
        ToolResultCommand(
          call.id,
          error: 'The person declined this call.',
          requestId: call.requestId,
          turnId: call.turnId,
        ),
      );
    }
  }

  /// Declines the sign-in that a call of one of the agent's connectors waits for, whose step in
  /// the conversation is `awaiting_authorization`. The router answers the call itself, so it
  /// runs nothing and the agent is told the person did not connect. A call that is not waiting
  /// is left alone.
  void declineLogin(String toolCallId) =>
      _held().send(ToolApprovalCommand(toolCallId, allowed: false));

  /// Puts the agent on this session's own call, `agent:<session id>`, and returns [session]
  /// as it now is.
  Future<Session> startVoice() async => _session = await _sessions.startVoice(id);

  /// Takes the agent off this session's call; the conversation goes on in writing.
  Future<Session> stopVoice() async => _session = await _sessions.stopVoice(id);

  /// Goes back to a response and carries on from there.
  ///
  /// The transcript shown here still has the later turns; read it back from [responses]
  /// after this if it should not.
  Future<void> rewind(String responseId) => responses.rewind(responseId);

  /// Renames or relabels this conversation, and returns [session] as it now is.
  ///
  /// A field left null is left as it is. [custom] replaces the labels whole, and an empty map
  /// clears them.
  Future<Session> update({
    String? title,
    String? description,
    Map<String, Object?>? custom,
  }) async =>
      _session = await _sessions.update(id, title: title, description: description, custom: custom);

  /// Continues this conversation as a new session, leaving this one as it was.
  ///
  /// The fork keeps this session's tools, since they are here in this process and a
  /// conversation continued without them would offer the model tools it cannot run. It is
  /// returned unstarted.
  Future<AgentSession> fork([ForkOptions options = const ForkOptions()]) async => AgentSession(
    await _sessions.fork(id, options),
    backend: _backend,
    sessions: _sessions,
    tools: _tools.values.toList(),
    interim: interim,
  );

  /// Stops the session and closes the socket. Safe to call more than once.
  ///
  /// What the conversation recorded and remembered is kept; [delete] takes it away.
  Future<void> close() async {
    final socket = _socket;
    if (isConnected && socket != null) {
      try {
        socket.send(const CloseCommand());
      } on SocketClosedException {
        // Gone already, which leaves the router to notice.
      }
    } else if (_connection.value is NotStarted) {
      await _sessions.stop(id);
    }
    await _hangUp();
  }

  /// Deletes this conversation, stopping it if it is running: its turns and what it taught
  /// memory go with it.
  Future<void> delete() async {
    await _sessions.delete(id);
    await _hangUp();
  }

  Future<void> _hangUp() async {
    await _socket?.close();
    await _pump?.cancel();
    _stopped(null);
  }

  void _asked(String text) {
    _conversation.value = _conversation.value.said(text.trim());
  }

  SessionSocket _held() {
    final socket = _socket;
    if (socket == null) {
      throw StateError('start() the session before talking to it');
    }
    return socket;
  }

  void _apply(AgentEvent event) {
    _conversation.value = _conversation.value.apply(event);
    switch (event.kind) {
      case AgentEventKind.toolCall:
        _answer(event.toolCall!, run: event['run'] != false);
      case AgentEventKind.toolCancel:
        final id = event['id'];
        if (id is String && _running.contains(id)) {
          _cancelled.add(id);
        }
        _approvals.value = [
          for (final request in _approvals.value)
            if (request.id != id) request,
        ];
      default:
        break;
    }
    for (final subscriber in [..._subscribers]) {
      if (subscriber.pending >= _bufferedEvents) {
        _subscribers.remove(subscriber);
        subscriber.controller.close();
        continue;
      }
      subscriber.pending++;
      subscriber.controller.add(event);
    }
  }

  /// Runs a tool the model asked for, or holds the call for the person when the tool asks
  /// first.
  ///
  /// [run] is false when the router says another watcher runs this call.
  void _answer(ToolCall call, {required bool run}) {
    // The request for frames is this device's only while its camera is on; otherwise a video
    // worker in the backend may own it.
    if (camera case final camera? when call.name == _cameraTool) {
      if (run) {
        _show(camera, call);
      }
      return;
    }
    // Session events go to observers as well as to whoever owns the tool. A worker elsewhere
    // may own this one, and an observer must not answer it first.
    final tool = _tools[call.name];
    if (tool == null) {
      return;
    }
    if (tool.approval case final approval?) {
      _approvals.value = [..._approvals.value, ToolApprovalRequest._(call, approval, tool)];
    } else {
      _run(call, tool);
    }
  }

  /// Answers the router's request for frames with pictures [camera] takes: what they are,
  /// then the pictures, oldest first. A camera with nothing to show is said as the answer
  /// rather than as a failure, as the router's own video workers say it, so the agent can ask
  /// the person to show it again. A request the router cancels is not answered.
  void _show(
    Future<List<({Uint8List jpeg, DateTime capturedAt})>> Function(DateTime at, int limit) camera,
    ToolCall call,
  ) {
    final arguments = call.argumentValues;
    final at = switch (arguments['at_ms']) {
      // Past what a DateTime holds is not a moment the router could have meant.
      final num ms when ms.abs() <= 8.64e15 => DateTime.fromMillisecondsSinceEpoch(ms.toInt()),
      _ => DateTime.now(),
    };
    final limit = switch (arguments['limit']) {
      final num limit => limit.toInt().clamp(1, _mostPictures),
      _ => 1,
    };
    _running.add(call.id);
    unawaited(() async {
      List<({Uint8List jpeg, DateTime capturedAt})> pictures;
      try {
        pictures = [...await camera(at, limit)]
          ..sort((a, b) => a.capturedAt.compareTo(b.capturedAt));
      } catch (_) {
        // A camera that could not take a picture has shown nothing.
        pictures = const [];
      }
      _running.remove(call.id);
      if (_cancelled.remove(call.id) || !isConnected) {
        return;
      }
      final frames = <Map<String, Object>>[];
      for (final picture in pictures) {
        _pictures += 1;
        frames.add({
          'captured_at_ms': picture.capturedAt.millisecondsSinceEpoch,
          'frame_id': 'camera-$_pictures',
        });
      }
      final output = pictures.isEmpty
          ? const ToolOutput(text: _nothingRecent)
          : ToolOutput(
              text: jsonEncode({'clock': 'device_unix_ms', 'frames': frames, 'source': 'camera'}),
              images: [
                for (final picture in pictures)
                  AgentImage('data:image/jpeg;base64,${base64Encode(picture.jpeg)}'),
              ],
            );
      try {
        _socket?.send(
          ToolResultCommand(
            call.id,
            output: output.text,
            images: output.images,
            requestId: call.requestId,
            turnId: call.turnId,
          ),
        );
      } on SocketClosedException {
        // The conversation ended while the camera was looking; there is nobody left to show.
      }
    }());
  }

  /// Runs a tool and sends back what it returned.
  ///
  /// On its own, not awaited, so a slow tool holds up neither the transcript nor the
  /// `tool_cancel` that might be on its way for it. A tool that throws is answered with the
  /// error: the model is waiting, and can only say something useful about a tool that did
  /// not work if it is told so.
  void _run(ToolCall call, AgentTool tool) {
    _running.add(call.id);
    unawaited(() async {
      ToolResultCommand result;
      try {
        final output = await tool.run(call.argumentValues);
        result = ToolResultCommand(
          call.id,
          output: output.text,
          images: output.images,
          requestId: call.requestId,
          turnId: call.turnId,
        );
      } catch (error) {
        result = ToolResultCommand(
          call.id,
          error: '$error',
          requestId: call.requestId,
          turnId: call.turnId,
        );
      }
      _running.remove(call.id);
      if (_cancelled.remove(call.id) || !isConnected) {
        return;
      }
      try {
        _socket?.send(result);
      } on SocketClosedException {
        // The conversation ended while the tool ran; there is nobody left to tell.
      }
    }());
  }

  void _stopped(AgentsException? failure) {
    if (_connection.value is Disconnected) {
      return;
    }
    _connection.value = Disconnected(failure);
    _conversation.value = _conversation.value.withState(ConversationState.ended);
    for (final subscriber in _subscribers) {
      subscriber.controller.close();
    }
    _subscribers.clear();
    _approvals.value = const [];
    _conversation.close();
    _connection.close();
    _approvals.close();
  }
}

/// The tool the router asks for frames with. Answered by the session, never offered to the
/// model as one of the app's tools.
const _cameraTool = 'get_video_frames';

/// The most pictures one request for frames is answered with.
const _mostPictures = 8;

/// What the agent is told when the camera has shown nothing recent enough.
const _nothingRecent =
    'The camera has shown nothing in the last few seconds; ask them to point it at what they '
    'mean again.';

/// A call of one of your tools, waiting for the person to allow it.
final class ToolApprovalRequest {
  ToolApprovalRequest._(this.call, this.approval, this._tool);

  final ToolCall call;

  /// The question the tool asks.
  final ToolApproval approval;
  final AgentTool _tool;

  String get id => call.id;

  /// The model's own words for why it wants the call, from the argument the approval names,
  /// or the empty string.
  String get reason => switch (approval.reasonArgument) {
    final name? => switch (call.argumentValues[name]) {
      final String text => text,
      _ => '',
    },
    null => '',
  };
}

final class _Subscriber {
  final controller = StreamController<AgentEvent>();
  int pending = 0;
}
