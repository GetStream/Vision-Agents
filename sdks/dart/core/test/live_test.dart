// Tests that need a router running.
//
// The Dart answer to @pytest.mark.integration: tagged `live`, and skipped unless
// VISION_AGENTS_URL is set, so the ordinary `dart test` stays offline and fast.
//
//     VISION_AGENTS_URL=http://localhost:8080 VISION_AGENTS_CUSTOMER_ID=examples \
//       VISION_AGENTS_AGENT=simple_voice_ai dart test -t live
@Tags(['live'])
library;

import 'dart:async';
import 'dart:io';

import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

final _url = Platform.environment['VISION_AGENTS_URL'];
final _customerId = Platform.environment['VISION_AGENTS_CUSTOMER_ID'] ?? 'acme';

/// An agent config name. Empty opens a session with no config, which the router answers
/// with its own defaults.
final _agent = Platform.environment['VISION_AGENTS_AGENT'] ?? '';

/// A user of our own per run, so what one run lists is not another's.
final _user = 'dart-live-${DateTime.now().millisecondsSinceEpoch}';

void main() {
  late VisionAgents agents;
  final opened = <AgentSession>[];

  Future<AgentSession> chat([SessionOptions options = const SessionOptions()]) async {
    final session = await agents.agent(_agent).chat(options);
    opened.add(session);
    return session;
  }

  setUp(() {
    agents = VisionAgents(url: Uri.parse(_url!), customerId: _customerId).withUser(_user);
  });

  tearDown(() async {
    for (final session in opened) {
      await session.close();
    }
    opened.clear();
    agents.close();
  });

  group(
    'live',
    () {
      test('a text session joins no call and opens a socket', () async {
        final session = await chat();

        expect(session.session.isText, isTrue);
        expect(session.session.callId, isEmpty);
        expect(session.session.state, SessionState.live);
        expect(session.isConnected, isTrue);
      });

      test('asking something gets an answer, one delta at a time', () async {
        final session = await chat();
        final deltas = session
            .events()
            .where((event) => event.kind == AgentEventKind.responseDelta)
            .first;

        session.send('What are your opening hours? Answer in one sentence.');

        await deltas.timeout(const Duration(seconds: 30));
        await _until(() => session.state == ConversationState.idle && session.turns.length >= 2);
        expect(session.turns.last.speaker.isAgent, isTrue);
        expect(session.turns.last.text, isNotEmpty);
        expect(session.failure, isNull);
      });

      test('a tool on this side is called and answered', () async {
        final asked = <String>[];
        final session = await chat(
          SessionOptions(
            tools: [
              AgentTool(
                name: 'lookup_order',
                description: "Look up one of the caller's orders by its order number.",
                parameters: AgentTool.strings(
                  {'order_id': 'the order number'},
                  required: ['order_id'],
                ),
                run: (arguments) async {
                  asked.add('${arguments['order_id']}');
                  return 'Order A-1042: 2 wool throws, 78.00, delivered 14 August, unopened.';
                },
              ),
            ],
          ),
        );

        session.send('Look up order A-1042 and tell me what is in it.');

        await _until(() => asked.isNotEmpty, seconds: 45);
        expect(asked.first.toUpperCase(), 'A-1042');
        await _until(() => session.state == ConversationState.idle && session.turns.length >= 2);
        expect(session.turns.last.text.toLowerCase(), contains('throw'));
      });

      test('a rewound session carries on from the response kept, and forks from it', () async {
        final session = await chat();
        session.send('My name is Ada. Reply with one word.');
        await _until(() => session.state == ConversationState.idle && session.turns.length >= 2);
        session.send('What is my name? Reply with one word.');
        await _until(() => session.state == ConversationState.idle && session.turns.length >= 4);
        await _until(() async => (await session.responses.list()).length == 2);

        final kept = (await session.responses.list()).first;
        expect(kept.said, contains('Ada'));

        await session.rewind(kept.id);
        expect((await session.responses.list()).map((r) => r.id), [kept.id]);

        final fork = await session.fork(ForkOptions(responseId: kept.id));
        opened.add(fork);
        expect(fork.id, isNot(session.id));
        expect((await agents.sessions.get(fork.id)).forkedFrom, session.id);
      });

      test('a response created over HTTP is read back item by item', () async {
        final session = await chat();

        final response = await session.responses.create(
          'Say the word "pineapple" and nothing else.',
        );

        await _until(() async {
          final items = await session.responses.items(responseId: response.id);
          return items.any((item) => item.kind == ItemKind.answer);
        }, seconds: 45);
        final items = await session.responses.unwind(responseId: response.id).toList();
        expect(items.first.kind, ItemKind.said);
        expect(items.map((item) => item.text).join(' ').toLowerCase(), contains('pineapple'));
      });

      test('a conversation is found again by what it was called', () async {
        final title = 'dart live ${DateTime.now().microsecondsSinceEpoch}';
        final session = await chat(SessionOptions(title: title, project: 'dart-sdk'));

        await _until(() async => (await agents.sessions.search(title)).isNotEmpty);
        final found = await agents.sessions.search(title);
        final listed = await agents
            .agent(_agent)
            .sessions
            .query(const SessionQuery(project: 'dart-sdk'));

        expect(found.first.id, session.id);
        expect(listed.map((s) => s.id), contains(session.id));
        expect((await agents.sessions.get(session.id)).title, title);
      });

      test('closing a session ends it for the router too', () async {
        final session = await chat();

        await session.close();

        // Only a running session is found by id, so an ended one is looked for among the closed.
        await _until(() async {
          final closed = await agents.sessions.query(
            const SessionQuery(state: SessionFilter.closed, limit: 50),
          );
          return closed.any((s) => s.id == session.id);
        });
      });

      test('a guest holds a conversation of its own', () async {
        final store = MemoryGuestStore();
        final guest = await agents.guestUser(name: 'Dart Guest', store: store);
        expect(guest.id, startsWith('guest-'));
        expect(guest.token, isNotEmpty);
        expect(await agents.guestUser(store: store), guest);

        final asGuest = agents.withGuest(guest);
        final session = await asGuest.agent(_agent).chat();
        opened.add(session);

        expect((await asGuest.sessions.query()).map((s) => s.id), contains(session.id));
        asGuest.close();
      });

      test('looking something up answers with sources', () async {
        final found = await agents.router().search(
          'What is the capital of France?',
          const SearchOptions(results: 2),
        );

        expect(found.provider, isNotEmpty);
        expect(found.results, isNotEmpty);
      });
    },
    skip: _url == null ? 'set VISION_AGENTS_URL to run against a router' : false,
    timeout: const Timeout(Duration(seconds: 90)),
  );
}

/// Runs until the condition holds, so a test waits for what the model does rather than for a
/// fixed number of seconds.
Future<void> _until(FutureOr<bool> Function() done, {int seconds = 30}) async {
  final deadline = DateTime.now().add(Duration(seconds: seconds));
  while (!await done()) {
    if (DateTime.now().isAfter(deadline)) {
      fail('gave up after ${seconds}s');
    }
    await Future<void>.delayed(const Duration(milliseconds: 200));
  }
}
