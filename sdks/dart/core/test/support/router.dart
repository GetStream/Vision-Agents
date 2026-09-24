import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// A request as it reached the far end.
final class Arrived {
  Arrived(this.method, this.uri, this.headers, this.body);

  final String method;
  final Uri uri;
  final HttpHeaders headers;
  final String body;

  String get path => uri.path;
  Map<String, String> get query => uri.queryParameters;
  Object? get json => body.isEmpty ? null : jsonDecode(body);
  String? header(String name) => headers.value(name);
}

/// An answer to script for a route.
final class Answer {
  const Answer(this.status, [this.body]);

  final int status;

  /// Encoded as JSON unless it is already a string.
  final Object? body;
}

/// A router on 127.0.0.1 that records what arrived and answers what a test scripted.
///
/// A real server rather than a stubbed client, so what the suite checks is the request this
/// SDK actually put on the wire: a header, a query parameter, an encoded path, a socket
/// upgrade.
final class TestRouter {
  TestRouter._(this._server) {
    _server.listen(_handle);
  }

  static Future<TestRouter> start() async =>
      TestRouter._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  final HttpServer _server;
  final Map<String, Answer> _answers = {};
  final List<Arrived> arrived = [];

  /// The socket upgrades that arrived, oldest first.
  final List<Arrived> upgrades = [];
  final List<TestSocket> _open = [];

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  /// Answers `METHOD /path` with this, until told otherwise.
  void answer(String route, Answer answer) => _answers[route] = answer;

  /// Answers every socket upgrade with a 404, the way a router that lost the session would.
  bool refuseSockets = false;

  /// The socket the SDK opened [index]th, once it has.
  Future<TestSocket> socket([int index = 0]) async {
    await until(() => _open.length > index);
    return _open[index];
  }

  Arrived last(String route) =>
      arrived.lastWhere((request) => '${request.method} ${request.path}' == route);

  Future<void> close() async {
    for (final socket in _open) {
      await socket.close(1001);
    }
    await _server.close(force: true);
  }

  Future<void> _handle(HttpRequest request) async {
    final body = await utf8.decoder.bind(request).join();
    final seen = Arrived(request.method, request.uri, request.headers, body);

    if (WebSocketTransformer.isUpgradeRequest(request)) {
      upgrades.add(seen);
      if (refuseSockets) {
        request.response.statusCode = 404;
        await request.response.close();
        return;
      }
      final socket = TestSocket(await WebSocketTransformer.upgrade(request));
      _open.add(socket);
      return;
    }

    arrived.add(seen);
    final answer =
        _answers['${request.method} ${request.uri.path}'] ??
        const Answer(404, {'error': 'no route'});
    request.response.statusCode = answer.status;
    if (answer.body != null) {
      request.response.headers.contentType = ContentType.json;
      request.response.write(answer.body is String ? answer.body : jsonEncode(answer.body));
    }
    await request.response.close();
  }
}

/// One socket the SDK opened, from the router's end.
final class TestSocket {
  TestSocket(this._socket) {
    _socket.listen(
      (message) => _received.add(jsonDecode(message as String) as Map<String, Object?>),
      onDone: () => _done.complete(),
    );
  }

  final WebSocket _socket;
  final _received = StreamController<Map<String, Object?>>();
  late final _frames = StreamIterator(_received.stream);
  final _done = Completer<void>();

  /// Sends a frame exactly as written, so a test can quote the router's `frameOf` verbatim.
  void send(String frame) => _socket.add(frame);

  /// The next frame the SDK sent.
  Future<Map<String, Object?>> next() async {
    if (!await _frames.moveNext().timeout(const Duration(seconds: 5))) {
      throw StateError('the socket closed before a frame arrived');
    }
    return _frames.current;
  }

  /// Resolves once the SDK has closed its end.
  Future<void> get done => _done.future.timeout(const Duration(seconds: 5));

  int? get closeCode => _socket.closeCode;

  Future<void> close(int code, [String reason = '']) => _socket.close(code, reason);
}

/// Runs until the condition holds, so a test waits for what happened rather than a guess.
Future<void> until(bool Function() done, {Duration within = const Duration(seconds: 5)}) async {
  final deadline = DateTime.now().add(within);
  while (!done()) {
    if (DateTime.now().isAfter(deadline)) {
      throw TimeoutException('gave up waiting', within);
    }
    await Future<void>.delayed(const Duration(milliseconds: 10));
  }
}
