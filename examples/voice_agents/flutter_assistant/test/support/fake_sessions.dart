import 'dart:async';
import 'dart:io' show WebSocket;

import 'package:flutter_assistant/agent_chat.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_agents/stream_agents.dart';

import 'fake_router.dart';

/// Sessions held by a router on this machine, each with a channel that is never fetched: what
/// a test says arrives in it is all it holds.
class FakeSessions extends AgentSessions {
  FakeSessions._(this._router)
    : _agents = StreamAgents(url: _router.url, customerId: 'acme', userId: 'ada');

  static Future<FakeSessions> onRouter() async => FakeSessions._(await FakeRouter.start());

  final FakeRouter _router;
  final StreamAgents _agents;
  final _client = stream.StreamChatClient('key');
  stream.Channel? _channel;

  /// The last channel a session was opened with.
  stream.Channel? get opened => _channel;

  /// What was said in the conversation before it is opened.
  List<stream.Message> history = const [];

  /// Holds the next session back until it completes.
  Completer<void>? hold;

  /// What starting a session fails with, or null.
  Object? startFailure;

  /// What opening a session's channel fails with, or null.
  Object? channelFailure;

  int started = 0;
  final deleted = <String>[];

  /// The requests the router took, each as `METHOD /path`, oldest first.
  List<String> get routes => _router.arrived;

  /// What the sessions' sockets sent, oldest first.
  List<String> get said => _router.said;

  /// Answers `METHOD /path` with [status] from now on.
  void answer(String route, int status) => _router.answer(route, status);

  /// Holds the answer to `METHOD /path` back until [until] completes.
  void holdRoute(String route, Future<void> until) => _router.hold(route, until);

  /// How many questions the router was asked.
  int get asked => _router.arrived.where((route) => route.endsWith('/responses')).length;

  /// How many times the router was told to stop a reply.
  int get interrupts => _router.said.where((message) => message.contains('"interrupt"')).length;

  /// Holds back the router's answer to the next question in session [id] until [until].
  void holdQuestion(String id, Future<void> until) =>
      _router.hold('POST /v1/agents/sessions/$id/responses', until);

  /// The open session's socket, from the router's end.
  Future<WebSocket> socket() => _router.socket();

  /// A message arriving in the open session's channel.
  void receive(stream.Message message) => _client.handleEvent(
    stream.Event(type: stream.EventType.messageNew, cid: _channel!.cid, message: message),
  );

  /// A message of the open session's channel changing.
  void update(stream.Message message) => _client.handleEvent(
    stream.Event(type: stream.EventType.messageUpdated, cid: _channel!.cid, message: message),
  );

  @override
  Future<AgentSession> start(String? sessionId) async {
    started++;
    await hold?.future;
    if (startFailure case final failure?) throw failure;
    final id = sessionId ?? 's$started';
    _router
      ..answer('POST /v1/agents/sessions', 201, sessionJson(id))
      ..answer('GET /v1/agents/sessions/$id', 200, sessionJson(id))
      ..answer('POST /v1/agents/sessions/$id/responses', 202, responseJson(id))
      ..answer('POST /v1/agents/sessions/$id/tools', 204)
      ..answer('DELETE /v1/agents/sessions/$id/tools/reserve_table', 204);
    return sessionId != null ? _agents.attach(sessionId) : _agents.chat();
  }

  @override
  Future<stream.Channel> channel(AgentSession session, {required int pageSize}) async {
    if (channelFailure case final failure?) throw failure;
    final channel = _channel = _client.channel('messaging', id: session.id);
    channel.state = stream.ChannelClientState(channel, stream.ChannelState(messages: history));
    return channel;
  }

  @override
  Future<void> delete(String sessionId) async => deleted.add(sessionId);

  Future<void> dispose() async {
    _agents.close();
    await _client.dispose();
    await _router.close();
  }
}
