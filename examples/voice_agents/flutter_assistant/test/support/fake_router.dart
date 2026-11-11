import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// A router on this machine that answers what a test scripted and holds the sessions' sockets,
/// so a test follows a session as the app does and nothing leaves the machine.
final class FakeRouter {
  FakeRouter._(this._server) {
    _server.listen(_handle);
  }

  static Future<FakeRouter> start() async =>
      FakeRouter._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  final HttpServer _server;
  final _answers = <String, (int, Object?)>{};
  final _sockets = <WebSocket>[];

  /// The requests that arrived, each as `METHOD /path`, oldest first.
  final arrived = <String>[];

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  /// Answers `METHOD /path` with this status and JSON body from now on.
  void answer(String route, int status, [Object? body]) => _answers[route] = (status, body);

  /// The socket of the session opened [index]th, from the router's end, once it is open.
  Future<WebSocket> socket([int index = 0]) async {
    await until(() => _sockets.length > index);
    return _sockets[index];
  }

  Future<void> close() async {
    for (final socket in _sockets) {
      await socket.close(1001);
    }
    await _server.close(force: true);
  }

  Future<void> _handle(HttpRequest request) async {
    if (WebSocketTransformer.isUpgradeRequest(request)) {
      final socket = await WebSocketTransformer.upgrade(request);
      // Read, so the socket notices the app closing its end.
      socket.listen((_) {}, onError: (_) {});
      _sockets.add(socket);
      return;
    }
    await request.drain<void>();
    final route = '${request.method} ${request.uri.path}';
    arrived.add(route);
    final (status, body) = _answers[route] ?? (404, _error('not_found', 'no route'));
    request.response.statusCode = status;
    if (body != null) {
      request.response.headers.contentType = ContentType.json;
      request.response.write(jsonEncode(body));
    }
    await request.response.close();
  }
}

/// A session as the router writes one.
Map<String, Object?> sessionJson(String id) => {
  'agent': 'docs',
  'agent_id': 'a1',
  'call_id': '',
  'call_type': 'agent',
  'config_id': 'c1',
  'conversation_id': 'messaging:$id',
  'created_at': '2026-09-24T09:54:56.038055123-06:00',
  'id': id,
  'instructions': 'Be brief.',
  'llm': 'gemini/gemini-3.8-flash',
  'modality': 'text',
  'mode': 'text',
  'state': 'live',
  'text': true,
  'user_id': 'ada',
};

/// A response as the router writes one once it has taken a question.
Map<String, Object?> responseJson(String sessionId) => {
  'id': 'r1',
  'session_id': sessionId,
  'said': 'Hi',
  'status': 'running',
  'created_at': '2026-09-24T15:54:56Z',
};

Map<String, Object?> _error(String code, String message) => {
  'error': {'message': message, 'type': code, 'code': code},
};

/// Waits until [done], so a test waits for what happened rather than for a guess.
Future<void> until(bool Function() done, {Duration within = const Duration(seconds: 5)}) async {
  final deadline = DateTime.now().add(within);
  while (!done()) {
    if (DateTime.now().isAfter(deadline)) throw TimeoutException('gave up waiting', within);
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}
