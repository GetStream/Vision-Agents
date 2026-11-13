import 'dart:async';
import 'dart:convert';
import 'dart:io';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_agents/stream_agents.dart';

/// A router on loopback that answers what a voice session does before joining a call:
/// creating and reading a session, and its events socket. Every request, body and command
/// sent over the socket is kept.
class _Router {
  _Router._(this._server) {
    _server.listen((request) async {
      arrived.add('${request.method} ${request.uri.path}');
      if (WebSocketTransformer.isUpgradeRequest(request)) {
        final socket = await WebSocketTransformer.upgrade(request);
        sockets.add(socket);
        socket.listen((frame) {
          final command = jsonDecode('$frame') as Map<String, Object?>;
          commands.add(command);
          // As the real router does once the session has ended.
          if (command['type'] == 'close') {
            socket.close(1000, 'the session ended');
          }
        });
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
  final sockets = <WebSocket>[];
  final arrived = <String>[];
  final bodies = <Map<String, Object?>>[];
  final commands = <Map<String, Object?>>[];

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  Future<void> close() async {
    for (final socket in sockets) {
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
    'modality': 'voice',
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
  // The widget binding answers every HTTP request 400; these talk to a router on loopback.
  TestWidgetsFlutterBinding.ensureInitialized();
  HttpOverrides.global = null;

  late _Router router;
  late StreamAgents agents;

  setUp(() async {
    router = await _Router.start();
    agents = StreamAgents(url: router.url, customerId: 'acme', userId: 'ada');
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
      expect(created['start_voice'], isTrue);
      expect(created.containsKey('call_id'), isFalse);
      expect(created.containsKey('text'), isFalse);
      expect(voice.session.isConnected, isTrue);
      expect(voice.state.value.isJoined, isFalse);
      await voice.end();
    });

    test('attaching reads the session rather than creating one', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      expect(voice.session.id, 'voice-1');
      expect(router.arrived.first, 'GET /v1/agents/sessions/voice-1');
      expect(router.bodies, isEmpty);
      await voice.end();
    });

    test('keeps why joining failed when the token could not be had', () async {
      final keyed = StreamAgents(
        url: router.url,
        customerId: 'acme',
        apiKey: 'stream-key',
        userId: 'ada',
        token: () async => throw const HttpException('backend unavailable'),
      );
      final voice = await VoiceSession.attach(keyed, 'voice-1');

      await voice.join();

      expect(voice.state.value.failure, isA<HttpException>());
      expect(voice.call, isNull);
      await voice.end();
      keyed.close();
    });

    test('joining needs the Stream app\'s key, which a customer id alone is not', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await expectLater(voice.join(), throwsArgumentError);

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

    test('turning the camera on before joining has no call to look through', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.setCameraEnabled(true);

      // ignore: invalid_use_of_internal_member
      expect(voice.session.camera, isNull);
      await voice.end();
    });

    test('leaving a session something else started leaves it running', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.leave();
      await Future<void>.delayed(const Duration(milliseconds: 50));

      expect(router.commands, isEmpty);
      expect(router.arrived, isNot(contains('POST /v1/agents/sessions/voice-1/stop')));
    });

    test('leaving a session something else started closes this device\'s socket', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.leave();

      await _until(() => router.sockets.single.closeCode != null);
      expect(voice.session.connection.value, const Disconnected());
    });

    test('hanging up after leaving a session something else started still ends it', () async {
      final voice = await VoiceSession.attach(agents, 'voice-1');

      await voice.leave();
      await voice.end();

      final askedToClose = router.commands.any((command) => command['type'] == 'close');
      final stopped = router.arrived.contains('POST /v1/agents/sessions/voice-1/stop');
      expect(askedToClose || stopped, isTrue);
    });

    test('says it is joining while it is, and over once it has left', () async {
      final asked = Completer<void>();
      final token = Completer<String>();
      final keyed = StreamAgents(
        url: router.url,
        customerId: 'acme',
        apiKey: 'stream-key',
        userId: 'ada',
        token: () {
          asked.complete();
          return token.future;
        },
      );
      final voice = await VoiceSession.attach(keyed, 'voice-1');

      final joining = voice.join();
      expect(voice.state.value.isJoining, isTrue);
      await asked.future;
      await voice.leave();
      token.completeError(const HttpException('backend unavailable'));
      await joining;

      expect(voice.state.value.isJoining, isFalse);
      expect(voice.state.value.isEnded, isTrue);
      expect(voice.state.value.failure, isNull);
      keyed.close();
    });

    test('joins nothing after hanging up', () async {
      final keyed = StreamAgents(
        url: router.url,
        customerId: 'acme',
        apiKey: 'stream-key',
        userId: 'ada',
        token: () async => throw const HttpException('backend unavailable'),
      );
      final voice = await VoiceSession.attach(keyed, 'voice-1');
      await voice.end();

      await voice.join();

      expect(voice.state.value.failure, isNull);
      expect(voice.state.value.isJoining, isFalse);
      keyed.close();
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

  group('VoiceCallView', () {
    testWidgets('says the call ended once hung up, not that it is still joining', (tester) async {
      final keyed = StreamAgents(
        url: router.url,
        customerId: 'acme',
        apiKey: 'stream-key',
        userId: 'ada',
        token: () => Completer<String>().future,
      );
      final voice = (await tester.runAsync(() => VoiceSession.attach(keyed, 'voice-1')))!;
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: VoiceCallView(voice: voice)),
        ),
      );
      await tester.pump();
      expect(find.text('joining'), findsOneWidget);

      await tester.runAsync(voice.end);
      await tester.pump();

      expect(find.text('joining'), findsNothing);
      expect(find.text('call ended'), findsOneWidget);
      expect(
        tester.widget<IconButton>(find.widgetWithIcon(IconButton, Icons.call_end)).onPressed,
        isNull,
      );
      keyed.close();
      await tester.pump(const Duration(seconds: 1));
    });
  });
}
