import 'dart:async';

import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'support/fixtures.dart';
import 'support/router.dart';

void main() {
  late TestRouter router;
  late VisionAgents agents;

  setUp(() async {
    router = await TestRouter.start();
    router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));
    agents = VisionAgents(url: router.url, customerId: 'acme');
  });

  tearDown(() async {
    agents.close();
    await router.close();
  });

  group('AgentSession', () {
    test('opens the socket as a device, with decisions off, before it returns', () async {
      final chat = await agents.withUser('ada').chat();

      expect(chat.isConnected, isTrue);
      final upgrade = router.upgrades.single;
      expect(upgrade.path, '/v1/agents/sessions/s1/events');
      expect(upgrade.query, {'decisions': 'false', 'customer_id': 'acme', 'user_id': 'ada'});
      expect(upgrade.header('Stream-Auth-Type'), isNull);
      await chat.close();
    });

    test('shows what was typed at once and sends it as a respond', () async {
      final chat = await agents.chat();
      final socket = await router.socket();

      chat.send('  What are your hours?  ');

      expect(chat.turns.single.text, 'What are your hours?');
      expect(await socket.next(), {'type': 'respond', 'text': 'What are your hours?'});
      await chat.close();
    });

    test('folds the router frames into the transcript and the state', () async {
      final chat = await agents.chat();
      final socket = await router.socket();

      socket
        ..send('{"type":"responding","turn_id":"t1","participant":{},"prompt":"hi"}')
        ..send('{"type":"response_delta","turn_id":"t1","text":"Nine"}')
        ..send(
          '{"pending_work":false,"type":"responded","turn_id":"t1","text":"Nine to five.","time_to_first_token_ms":90}',
        );

      await until(() => chat.turns.isNotEmpty && chat.state == ConversationState.idle);
      expect(chat.turns.single.text, 'Nine to five.');
      await chat.close();
    });

    test('a widget building from the conversation sees each change as it lands', () async {
      final chat = await agents.chat();
      final socket = await router.socket();
      final states = <ConversationState>[];
      final subscription = chat.conversation.stream.listen((c) => states.add(c.state));

      socket.send('{"type":"responding","turn_id":"t1"}');
      socket.send('{"type":"responded","turn_id":"t1","text":"Hi."}');

      await until(() => states.length >= 3);
      expect(states, [
        ConversationState.idle,
        ConversationState.responding,
        ConversationState.idle,
      ]);
      await subscription.cancel();
      await chat.close();
    });

    test('skips a frame it cannot read rather than ending the conversation', () async {
      final chat = await agents.chat();
      final socket = await router.socket();

      socket
        ..send('not json')
        ..send('{"no":"type"}')
        ..send('{"type":"response_delta","turn_id":"t1","text":"still here"}');

      await until(() => chat.turns.isNotEmpty);
      expect(chat.turns.single.text, 'still here');
      expect(chat.isConnected, isTrue);
      await chat.close();
    });

    test('runs a tool on this side and answers the model with what it returned', () async {
      final asked = <Object?>[];
      final chat = await agents.chat(
        SessionOptions(
          tools: [
            AgentTool(
              name: 'lookup_order',
              description: 'Look up an order.',
              run: (arguments) async {
                asked.add(arguments['order_id']);
                return 'Order A-1042: 2 wool throws.';
              },
            ),
          ],
        ),
      );
      final socket = await router.socket();

      socket.send(
        '{"type":"tool_call","id":"c1","name":"lookup_order",'
        r'"arguments":"{\"order_id\":\"A-1042\"}","command_id":"","turn_id":"t1"}',
      );

      expect(await socket.next(), {
        'type': 'tool_result',
        'tool_call_id': 'c1',
        'output': 'Order A-1042: 2 wool throws.',
        'error': '',
      });
      expect(asked, ['A-1042']);
      await chat.close();
    });

    test('tells the model a tool did not work, since it is mid-sentence waiting', () async {
      final chat = await agents.chat(
        SessionOptions(
          tools: [
            AgentTool(
              name: 'lookup_order',
              description: 'Look up an order.',
              run: (_) async => throw StateError('the order service is down'),
            ),
          ],
        ),
      );
      final socket = await router.socket();

      socket.send('{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"{}"}');

      final result = await socket.next();
      expect(result['output'], '');
      expect(result['error'], contains('the order service is down'));
      await chat.close();
    });

    test("repeats a durable command's ids, so no other command can adopt the result", () async {
      final chat = await agents.chat(
        SessionOptions(
          tools: [AgentTool(name: 'now', description: 'The time.', run: (_) async => 'noon')],
        ),
      );
      final socket = await router.socket();

      socket.send(
        '{"type":"tool_call","id":"c1","name":"now","arguments":"{}","command_id":"cmd-1","turn_id":"t1"}',
      );

      expect(await socket.next(), {
        'type': 'tool_result',
        'tool_call_id': 'c1',
        'output': 'noon',
        'error': '',
        'command_id': 'cmd-1',
        'turn_id': 't1',
      });
      await chat.close();
    });

    test('leaves a tool it does not have for whoever does, rather than answering first', () async {
      final chat = await agents.chat(
        SessionOptions(
          tools: [AgentTool(name: 'mine', description: 'Mine.', run: (_) async => 'mine')],
        ),
      );
      final socket = await router.socket();

      socket
        ..send('{"type":"tool_call","id":"c1","name":"somebody_elses","arguments":"{}"}')
        ..send('{"type":"tool_call","id":"c2","name":"mine","arguments":"{}"}');

      expect((await socket.next())['tool_call_id'], 'c2');
      await chat.close();
    });

    test('a tool call the router cancelled is not answered', () async {
      final release = Completer<String>();
      final chat = await agents.chat(
        SessionOptions(
          tools: [
            AgentTool(name: 'slow', description: 'Slow.', run: (_) => release.future),
            AgentTool(name: 'fast', description: 'Fast.', run: (_) async => 'fast'),
          ],
        ),
      );
      final socket = await router.socket();
      final cancelled = chat.events().firstWhere(
        (event) => event.kind == AgentEventKind.toolCancel,
      );

      socket
        ..send('{"type":"tool_call","id":"c1","name":"slow","arguments":"{}"}')
        ..send('{"type":"tool_cancel","id":"c1","command_id":"","turn_id":""}');
      await cancelled;
      release.complete('too late');
      socket.send('{"type":"tool_call","id":"c2","name":"fast","arguments":"{}"}');

      expect((await socket.next())['tool_call_id'], 'c2');
      await chat.close();
    });

    test('answering a tool does not depend on anybody reading events', () async {
      final chat = await agents.chat(
        SessionOptions(
          tools: [AgentTool(name: 'now', description: 'The time.', run: (_) async => 'noon')],
        ),
      );
      final socket = await router.socket();

      socket.send('{"type":"tool_call","id":"c1","name":"now","arguments":"{}"}');

      expect((await socket.next())['output'], 'noon');
      await chat.close();
    });

    test('two readers of events each get every event', () async {
      final chat = await agents.chat();
      final socket = await router.socket();
      final first = chat.events().map((event) => event.type).take(2).toList();
      final second = chat.events().map((event) => event.type).take(2).toList();

      socket
        ..send('{"type":"joined","at":"2026-09-24T15:54:56Z"}')
        ..send('{"type":"astonished"}');

      expect(await first, ['joined', 'astonished']);
      expect(await second, ['joined', 'astonished']);
      await chat.close();
    });

    test('a reader that falls far behind is finished rather than buffered forever', () async {
      final chat = await agents.chat();
      final socket = await router.socket();
      final slow = chat.events();

      for (var i = 0; i < 300; i++) {
        socket.send('{"type":"response_delta","turn_id":"t1","text":"."}');
      }
      await until(() => chat.turns.isNotEmpty && chat.turns.single.text.length == 300);

      expect(await slow.length, 256);
      expect(chat.isConnected, isTrue);
      await chat.close();
    });

    test('a session the router ended is over, with no failure to report', () async {
      final chat = await agents.chat();
      final socket = await router.socket();

      socket.send('{"type":"left","at":"2026-09-24T15:54:56Z"}');
      await socket.close(1000, 'the session ended');

      await until(() => chat.connection.value is Disconnected);
      expect(chat.failure, isNull);
      expect(chat.state, ConversationState.ended);
    });

    test('a socket lost mid-conversation reports why, with its close code', () async {
      final chat = await agents.chat();
      final socket = await router.socket();

      await socket.close(4000, 'draining');

      await until(() => chat.connection.value is Disconnected);
      expect(
        chat.failure,
        isA<SocketClosedException>()
            .having((e) => e.code, 'code', 4000)
            .having((e) => e.reason, 'reason', 'draining'),
      );
      expect(() => chat.send('anyone?'), throwsA(isA<SocketClosedException>()));
    });

    test('closing says so over the socket and is safe to repeat', () async {
      final chat = await agents.chat();
      final socket = await router.socket();

      await chat.close();
      await chat.close();

      expect(await socket.next(), {'type': 'close'});
      await socket.done;
      expect(socket.closeCode, 1000);
      expect(chat.connection.value, const Disconnected());
    });

    test(
      'a session whose socket cannot open is ended rather than left holding the agent',
      () async {
        router
          ..answer('DELETE /v1/agents/sessions/s1', const Answer(204))
          ..refuseSockets = true;

        await expectLater(agents.chat(), throwsA(isA<TransportException>()));
        expect(router.last('DELETE /v1/agents/sessions/s1').method, 'DELETE');
      },
    );

    test('a fork keeps the tools, since they are here in this process', () async {
      router.answer('POST /v1/agents/sessions/s1/fork', Answer(201, sessionJson(id: 's2')));
      final chat = await agents.chat(
        SessionOptions(
          tools: [AgentTool(name: 'now', description: 'The time.', run: (_) async => 'noon')],
        ),
      );

      final fork = await chat.fork(const ForkOptions(responseId: 'r1'));
      await fork.start();
      final socket = await router.socket(1);
      socket.send('{"type":"tool_call","id":"c1","name":"now","arguments":"{}"}');

      expect(fork.id, 's2');
      expect((await socket.next())['output'], 'noon');
      await fork.close();
      await chat.close();
    });
  });
}
