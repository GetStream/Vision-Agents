import 'dart:async';

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

  final Backend _backend;
  final Sessions _sessions;
  final Map<String, AgentTool> _tools;
  final _conversation = LiveValueController(const Conversation());
  final _connection = LiveValueController<Connection>(Connection.notStarted);
  final Set<_Subscriber> _subscribers = {};
  final Set<String> _running = {};
  final Set<String> _cancelled = {};
  SessionSocket? _socket;
  StreamSubscription<AgentEvent>? _pump;

  /// What the router holds this session by, which addresses it and its socket.
  String get id => session.id;

  /// The transcript and what the agent is doing.
  LiveValue<Conversation> get conversation => _conversation;

  /// Whether the socket is carrying the conversation, and why it stopped.
  LiveValue<Connection> get connection => _connection;

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
        _answer(event.toolCall!);
      case AgentEventKind.toolCancel:
        final id = event['id'];
        if (id is String && _running.contains(id)) {
          _cancelled.add(id);
        }
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

  /// Runs a tool the model asked for and sends back what it returned.
  ///
  /// On its own, not awaited, so a slow tool holds up neither the transcript nor the
  /// `tool_cancel` that might be on its way for it. A tool that throws is answered with the
  /// error: the model is waiting, and can only say something useful about a tool that did
  /// not work if it is told so.
  void _answer(ToolCall call) {
    // Session events go to observers as well as to whoever owns the tool. A worker elsewhere
    // may own this one, and an observer must not answer it first.
    final tool = _tools[call.name];
    if (tool == null) {
      return;
    }
    _running.add(call.id);
    unawaited(() async {
      ToolResultCommand result;
      try {
        final output = await tool.run(call.argumentValues);
        result = ToolResultCommand(
          call.id,
          output: output,
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
    _conversation.close();
    _connection.close();
  }
}

final class _Subscriber {
  final controller = StreamController<AgentEvent>();
  int pending = 0;
}
