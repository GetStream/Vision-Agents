import 'dart:convert';
import 'dart:io';

import 'package:flutter_assistant/settings.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('ReplyFormat', () {
    late _FakeBackend backend;

    setUp(() async => backend = await _FakeBackend.start());
    tearDown(() => backend.close());

    test('current reads the format the backend says the agent answers in', () async {
      for (final format in ReplyFormat.values) {
        backend.answer(200, jsonEncode({'format': format.name}));

        expect(await ReplyFormat.current(backend: backend.url), format);
      }
    });

    test('current is none for a format the app does not offer', () async {
      backend.answer(200, jsonEncode({'format': 'openui'}));

      expect(await ReplyFormat.current(backend: backend.url), isNull);
    });

    test('choose puts the format to the backend', () async {
      backend.answer(200, jsonEncode({'format': 'a2ui'}));

      await ReplyFormat.a2ui.choose(backend: backend.url);

      expect(backend.arrived, [
        'PUT /reply-format ${jsonEncode({'format': 'a2ui'})}',
      ]);
    });

    test('fails with what the backend said when it refuses', () async {
      backend.answer(400, 'no such format');

      await expectLater(
        ReplyFormat.markdown.choose(backend: backend.url),
        throwsA(isA<Exception>().having((e) => '$e', 'message', contains('no such format'))),
      );
    });
  });
}

/// The swift_assistant backend on this machine, answering what a test scripted.
final class _FakeBackend {
  _FakeBackend._(this._server) {
    _server.listen(_handle);
  }

  static Future<_FakeBackend> start() async =>
      _FakeBackend._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  final HttpServer _server;
  var _answer = (404, '');

  /// The requests that arrived, each as `METHOD /path body`, oldest first.
  final arrived = <String>[];

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  /// Answers every request with this status and body from now on.
  void answer(int status, String body) => _answer = (status, body);

  Future<void> close() => _server.close(force: true);

  Future<void> _handle(HttpRequest request) async {
    final body = await utf8.decodeStream(request);
    arrived.add('${request.method} ${request.uri.path} $body'.trim());
    final (status, answer) = _answer;
    request.response
      ..statusCode = status
      ..write(answer);
    await request.response.close();
  }
}
