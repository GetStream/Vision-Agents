import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';
import 'package:vision_agents_rtc/vision_agents_rtc.dart';

/// A router on loopback that answers what a voice session does before joining a call:
/// creating and reading a session, and its events socket. Every request, body and command
/// sent over the socket is kept.
class _Router {
  _Router._(this._server) {
    _server.listen((request) async {
      arrived.add('${request.method} ${request.uri.path}');
      if (WebSocketTransformer.isUpgradeRequest(request)) {
        final socket = await WebSocketTransformer.upgrade(request);
        _sockets.add(socket);
        socket.listen((frame) => commands.add(jsonDecode('$frame') as Map<String, Object?>));
        return;
      }
      final body = await utf8.decodeStream(request);
      if (body.isNotEmpty) {
        bodies.add(jsonDecode(body) as Map<String, Object?>);
      }
      request.response.headers.contentType = ContentType.json;
      request.response.write(jsonEncode(_session));
      await request.response.close();
    });
  }

  static Future<_Router> start() async =>
      _Router._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  final HttpServer _server;
  final _sockets = <WebSocket>[];
  final arrived = <String>[];
  final bodies = <Map<String, Object?>>[];
  final commands = <Map<String, Object?>>[];

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  Future<void> close() async {
    for (final socket in _sockets) {
      await socket.close();
    }
    await _server.close(force: true);
  }

  static const _session = {
    'id': 'voice-1',
    'call_id': 'call-1',
    'call_type': 'default',
    'user_id': 'vision-agent',
    'agent_id': 'a1',
    'created_at': '2026-09-24T09:54:56.038055-06:00',
    'state': 'live',
    'mode': 'cascade',
  };
}

Future<void> _until(bool Function() done) async {
  final deadline = DateTime.now().add(const Duration(seconds: 5));
  while (!done()) {
    if (DateTime.now().isAfter(deadline)) {
      fail('gave up waiting');
    }
    await Future<void>.delayed(const Duration(milliseconds: 10));
  }
}

void main() {
  late _Router router;
  late VisionAgents agents;

  setUp(() async {
    router = await _Router.start();
    agents = VisionAgents(url: router.url, customerId: 'acme', userId: 'ada');
  });

  tearDown(() async {
    agents.close();
    await router.close();
  });

  group('VoiceSession', () {
    test('starting puts the agent on a call of its own and follows it', () async {
      final voice = await VoiceSession.start(agents, agent: 'support');

      final created = router.bodies.single;
      expect(created['agent'], 'support');
      expect('${created['call_id']}', matches(RegExp(r'^[0-9a-f]{32}$')));
      expect(created.containsKey('text'), isFalse);
      expect(voice.session.isConnected, isTrue);
      expect(voice.state.value.isJoined, isFalse);
      await voice.end();
    });

    test('starting on a call that is named uses that call', () async {
      final voice = await VoiceSession.start(agents, callId: 'front-door');

      expect(router.bodies.single['call_id'], 'front-door');
      await voice.end();
    });

    test('attaching reads the session rather than creating one', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      expect(voice.session.id, 'voice-1');
      expect(router.arrived.first, 'GET /v1/agents/sessions/voice-1');
      expect(router.bodies, isEmpty);
      await voice.end();
    });

    test('keeps why joining failed, having asked for credentials for this session', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');
      final asked = <String>[];

      await voice.join(
        credentials: (sessionId) async {
          asked.add(sessionId);
          throw const HttpException('backend unavailable');
        },
      );

      expect(asked, ['voice-1']);
      expect(voice.state.value.failure, isA<HttpException>());
      expect(voice.call, isNull);
      await voice.end();
    });

    test('leaves the microphone and camera alone before joining', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.setMuted(true);
      await voice.setCameraEnabled(true);

      expect(voice.state.value.isMuted, isFalse);
      expect(voice.state.value.isCameraEnabled, isFalse);
      await voice.end();
    });

    test('leaving a session something else started leaves it running', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.leave();
      await Future<void>.delayed(const Duration(milliseconds: 50));

      expect(router.commands, isEmpty);
    });

    test('leaving a session this device started ends it', () async {
      final voice = await VoiceSession.start(agents);

      await voice.leave();

      await _until(() => router.commands.isNotEmpty);
      expect(router.commands.single['type'], 'close');
    });

    test('hanging up ends the session, whoever started it', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.end();

      await _until(() => router.commands.isNotEmpty);
      expect(router.commands.single['type'], 'close');
    });
  });
}
