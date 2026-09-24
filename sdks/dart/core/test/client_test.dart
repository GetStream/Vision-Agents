import 'dart:io';

import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'support/fixtures.dart';
import 'support/router.dart';

void main() {
  late TestRouter router;
  late VisionAgents agents;

  setUp(() async {
    router = await TestRouter.start();
    agents = VisionAgents(url: router.url, customerId: 'acme');
  });

  tearDown(() async {
    agents.close();
    await router.close();
  });

  group('Backend', () {
    test('declares a device on every request, even to a router that believes the header', () async {
      router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));

      await agents.withUser('ada').sessions.get('s1');

      final request = router.last('GET /v1/agents/sessions/s1');
      expect(request.header('X-Customer-Id'), 'acme');
      expect(request.header('Stream-Auth-Type'), 'jwt');
      expect(request.header('X-Stream-User-Id'), 'ada');
      expect(request.header('Authorization'), isNull);
    });

    test('presents a token with the api key, asking the provider each time', () async {
      router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
      var asked = 0;
      final keyed = VisionAgents(
        url: router.url,
        apiKey: 'key',
        token: () async => 'token-${++asked}',
      );

      await keyed.sessions.get('s1');
      await keyed.sessions.get('s1');
      keyed.close();

      final request = router.last('GET /v1/agents/sessions/s1');
      expect(request.header('X-Api-Key'), 'key');
      expect(request.header('Authorization'), 'Bearer token-2');
      expect(request.header('Stream-Auth-Type'), 'jwt');
      expect(request.header('X-Customer-Id'), isNull);
    });

    test('refuses an api key with no token before sending anything', () {
      expect(() => Backend(url: Uri.parse('http://x'), apiKey: 'key'), throwsArgumentError);
      expect(() => Backend(url: Uri.parse('http://x')), throwsArgumentError);
    });

    test('puts credentials in a socket url and no auth type, which has no query form', () async {
      final backend = Backend(
        url: Uri.parse('https://router.example/base'),
        customerId: 'acme',
      ).withUser('ada');

      final url = await backend.socketUri('/v1/agents/sessions/s1/events', {'decisions': 'false'});

      expect(url.scheme, 'wss');
      expect(url.path, '/base/v1/agents/sessions/s1/events');
      expect(url.queryParameters, {'decisions': 'false', 'customer_id': 'acme', 'user_id': 'ada'});
    });
  });

  group('Sessions', () {
    test('opens a written conversation, sending only what the caller set', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      final session = await agents.sessions.create(
        const SessionOptions(
          agent: 'docs',
          title: 'Billing',
          modelOverwrites: ModelOverwrites(thinking: Thinking.high),
        ),
      );

      expect(router.last('POST /v1/agents/sessions').json, {
        'text': true,
        'agent': 'docs',
        'title': 'Billing',
        'model_overwrites': {'thinking': 'high'},
      });
      expect(session.id, 's1');
      expect(session.isText, isTrue);
      expect(session.mode, SessionMode.text);
      expect(session.createdAt, DateTime.utc(2026, 9, 24, 15, 54, 56, 38, 55));
    });

    test('a voice session names its call and is not marked text', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.sessions.create(const SessionOptions(configId: 'c1'), 'call-1');

      expect(router.last('POST /v1/agents/sessions').json, {
        'call_id': 'call-1',
        'config_id': 'c1',
      });
    });

    test('declares tools by name, description and schema, and never their handler', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.sessions.create(
        SessionOptions(
          tools: [
            AgentTool(
              name: 'lookup_order',
              description: 'Look up an order.',
              parameters: AgentTool.strings({'order_id': 'the number'}, required: ['order_id']),
              run: (_) async => '',
            ),
          ],
        ),
      );

      expect((router.last('POST /v1/agents/sessions').json as Map)['tools'], [
        {
          'name': 'lookup_order',
          'description': 'Look up an order.',
          'parameters': {
            'type': 'object',
            'properties': {
              'order_id': {'type': 'string', 'description': 'the number'},
            },
            'required': ['order_id'],
          },
        },
      ]);
    });

    test("an agent's sessions are opened against it and listed by it", () async {
      router
        ..answer('POST /v1/agents/sessions', Answer(201, sessionJson()))
        ..answer('GET /v1/agents/sessions', Answer(200, [sessionJson()]));
      final docs = agents.agent('docs');

      await docs.sessions.create();
      await docs.sessions.query(
        SessionQuery(
          state: SessionFilter.closed,
          custom: {'tenant': 'acme'},
          createdAfter: DateTime.utc(2026, 9, 1),
          limit: 10,
        ),
      );

      expect(router.last('POST /v1/agents/sessions').json, {'text': true, 'agent': 'docs'});
      expect(router.last('GET /v1/agents/sessions').query, {
        'agent': 'docs',
        'state': 'closed',
        'custom': '{"tenant":"acme"}',
        'created_after': '2026-09-01T00:00:00.000Z',
        'limit': '10',
      });
    });

    test('naming a config by id does not also send the agent it is scoped to', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.agent('docs').sessions.create(const SessionOptions(configId: 'c2'));

      expect(router.last('POST /v1/agents/sessions').json, {'text': true, 'config_id': 'c2'});
    });

    test('searches by the words a conversation was titled with', () async {
      router.answer('GET /v1/agents/sessions/search', Answer(200, [sessionJson(id: 's2')]));

      final found = await agents.sessions.search(
        "o'brien billing",
        const SessionQuery(project: 'Health'),
      );

      expect(found.single.id, 's2');
      expect(router.last('GET /v1/agents/sessions/search').query, {
        'q': "o'brien billing",
        'project': 'Health',
      });
    });

    test('encodes a session id into the path rather than trusting it', () async {
      router.answer('DELETE /v1/agents/sessions/a%2Fb', const Answer(204));

      await agents.sessions.close('a/b');

      expect(router.arrived.single.path, '/v1/agents/sessions/a%2Fb');
    });

    test('forks at a response, sending history false only when it was asked for', () async {
      router.answer('POST /v1/agents/sessions/s1/fork', Answer(201, sessionJson(id: 's2')));

      final fork = await agents.sessions.fork('s1', const ForkOptions(responseId: 'r1'));
      await agents.sessions.fork('s1', const ForkOptions(withoutHistory: true, title: 'Again'));

      expect(fork.id, 's2');
      expect(router.arrived[0].json, {'response_id': 'r1'});
      expect(router.arrived[1].json, {'title': 'Again', 'messages': false});
    });

    test('reads a state it has never heard of as unknown rather than failing', () async {
      router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson(state: 'hibernating')));

      final session = await agents.sessions.get('s1');

      expect(session.state, SessionState.unknown);
    });

    test('says which field it could not read', () async {
      router.answer('GET /v1/agents/sessions/s1', Answer(200, {...sessionJson(), 'id': 7}));

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(isA<UnreadableException>().having((e) => e.what, 'what', contains('Session.id'))),
      );
    });

    test("reports the router's own words and status, not a status phrase", () async {
      router.answer('GET /v1/agents/sessions/s1', const Answer(404, {'error': 'no such session'}));

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(
          isA<RouterException>()
              .having((e) => e.status, 'status', 404)
              .having((e) => e.message, 'message', 'no such session')
              .having((e) => e.operation, 'operation', 'getSession'),
        ),
      );
    });

    test('a 403 is a server-side only path, and says so', () async {
      router.answer('GET /v1/agents/sessions/s1', const Answer(403, {'error': 'server side only'}));

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(isA<RouterException>().having((e) => e.isServerSideOnly, 'isServerSideOnly', true)),
      );
    });

    test('keeps what a proxy said when the refusal is not JSON', () async {
      router.answer('GET /v1/agents/sessions/s1', const Answer(502, 'Bad Gateway'));

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(isA<RouterException>().having((e) => e.message, 'message', 'Bad Gateway')),
      );
    });

    test('a router nobody is listening on is a transport failure, not a status', () async {
      final nowhere = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      final port = nowhere.port;
      await nowhere.close();
      final unreachable = VisionAgents(
        url: Uri.parse('http://127.0.0.1:$port'),
        customerId: 'acme',
      );

      await expectLater(unreachable.sessions.get('s1'), throwsA(isA<TransportException>()));
      unreachable.close();
    });
  });

  group('Responses', () {
    test('asks something and gets a handle on the turn answering it', () async {
      router.answer(
        'POST /v1/agents/sessions/s1/responses',
        Answer(202, responseJson('r1', status: 'running')),
      );

      final response = await agents.sessions
          .responses('s1')
          .create(
            'What is my name?',
            images: [const AgentImage('https://x/cat.png', detail: 'low')],
          );

      expect(response.id, 'r1');
      expect(response.status, ResponseStatus.running);
      expect(router.last('POST /v1/agents/sessions/s1/responses').json, {
        'text': 'What is my name?',
        'images': [
          {'url': 'https://x/cat.png', 'detail': 'low'},
        ],
      });
    });

    test('lists the turns, oldest first, with their finish times', () async {
      router.answer(
        'GET /v1/agents/sessions/s1/responses',
        Answer(200, [responseJson('r1'), responseJson('r2', status: 'cancelled')]),
      );

      final turns = await agents.sessions.responses('s1').list(limit: 5);

      expect(turns.map((turn) => turn.id), ['r1', 'r2']);
      expect(turns.last.status, ResponseStatus.cancelled);
      expect(turns.first.finishedAt, DateTime.utc(2026, 9, 24, 15, 54, 57, 500));
      expect(router.last('GET /v1/agents/sessions/s1/responses').query, {'limit': '5'});
    });

    test('unwinds every item a page at a time, stopping at a short page', () async {
      Map<String, Object?> item(int ordinal) => {
        'response_id': 'r1',
        'ordinal': ordinal,
        'kind': ordinal == 0 ? 'said' : 'tool_call',
        'tool_name': 'lookup_order',
        'at': '2026-09-24T15:54:56Z',
      };
      router.answer('GET /v1/agents/sessions/s1/responses/items', Answer(200, [item(0), item(1)]));

      final items = await agents.sessions
          .responses('s1')
          .unwind(responseId: 'r1', pageSize: 3)
          .toList();

      expect(items.map((item) => item.kind), [ItemKind.said, ItemKind.toolCall]);
      expect(router.arrived, hasLength(1));
      expect(router.last('GET /v1/agents/sessions/s1/responses/items').query, {
        'response_id': 'r1',
        'limit': '3',
        'offset': '0',
      });
    });

    test('rewinds to a response, which answers nothing', () async {
      router.answer('POST /v1/agents/sessions/s1/rewind', const Answer(204));

      await agents.sessions.responses('s1').rewind('r1');

      expect(router.last('POST /v1/agents/sessions/s1/rewind').json, {'response_id': 'r1'});
    });

    test('a conversation kept in chat cannot be rewound, and the router says to fork it', () async {
      router.answer(
        'POST /v1/agents/sessions/s1/rewind',
        const Answer(400, {
          'error': 'a persistent conversation cannot be rewound; fork it at the response',
        }),
      );

      await expectLater(
        agents.sessions.responses('s1').rewind('r1'),
        throwsA(isA<RouterException>().having((e) => e.message, 'message', contains('fork'))),
      );
    });

    test('refuses to rewind to a response that was never recorded', () {
      expect(() => agents.sessions.responses('s1').rewind(''), throwsArgumentError);
    });
  });

  group('VisionAgents.guestUser', () {
    Map<String, Object?> guest(String id, {String token = 't1', DateTime? expires}) => {
      'id': id,
      'token': token,
      'name': 'Guest',
      'expires_at': (expires ?? DateTime.now().add(const Duration(days: 1)))
          .toUtc()
          .toIso8601String(),
    };

    test('mints a guest and remembers it, so coming back is the same person', () async {
      router.answer('POST /v1/agents/guests', Answer(201, guest('guest-1')));
      final store = MemoryGuestStore();

      final first = await agents.guestUser(name: 'Guest', store: store);
      final again = await agents.guestUser(store: store);

      expect(first.id, 'guest-1');
      expect(again, first);
      expect(router.arrived, hasLength(1));
      expect(router.arrived.single.json, {'name': 'Guest'});
    });

    test('asks for a fresh token under the same id when the remembered one expired', () async {
      router.answer('POST /v1/agents/guests', Answer(201, guest('guest-1', token: 't2')));
      final store = MemoryGuestStore();
      await store.write(
        '{"id":"guest-1","token":"t1","expires_at":"${DateTime.now().subtract(const Duration(hours: 1)).toUtc().toIso8601String()}"}',
      );

      final renewed = await agents.guestUser(store: store);

      expect(renewed.token, 't2');
      expect(router.arrived.single.json, {'id': 'guest-1'});
    });

    test('a fresh guest ignores the one remembered, for a "not me" button', () async {
      router.answer('POST /v1/agents/guests', Answer(201, guest('guest-2')));
      final store = MemoryGuestStore();
      await store.write('{"id":"guest-1","token":"t1"}');

      final other = await agents.guestUser(store: store, fresh: true);

      expect(other.id, 'guest-2');
      expect(router.arrived.single.json, <String, Object?>{});
    });

    test('something unreadable under the key mints a guest rather than failing', () async {
      router.answer('POST /v1/agents/guests', Answer(201, guest('guest-3')));
      final store = MemoryGuestStore();
      await store.write('{truncated');

      expect((await agents.guestUser(store: store)).id, 'guest-3');
    });

    test('asks as the guest afterwards', () async {
      router
        ..answer('POST /v1/agents/guests', Answer(201, guest('guest-1')))
        ..answer('GET /v1/agents/sessions', const Answer(200, []));

      final minted = await agents.guestUser();
      await agents.withGuest(minted).sessions.query();

      expect(router.last('GET /v1/agents/sessions').header('X-Stream-User-Id'), 'guest-1');
    });
  });

  group('SearchRouter', () {
    test('asks under a config and gets the sources back', () async {
      router.answer(
        'POST /v1/search',
        const Answer(200, {
          'provider': 'exa',
          'model': 'exa-fast',
          'answer': 'Cefazolin.',
          'results': [
            {'url': 'https://x', 'title': 'Guide', 'score': 0.5},
          ],
        }),
      );

      final found = await agents
          .router(config: 'healthcare')
          .search('antibiotic guidance', const SearchOptions(depth: SearchDepth.fast, results: 3));

      expect(found.answer, 'Cefazolin.');
      expect(found.results.single.score, 0.5);
      expect(router.last('POST /v1/search').json, {
        'config_id': 'healthcare',
        'query': 'antibiotic guidance',
        'options': {'depth': 'fast', 'results': 3},
      });
    });

    test('sends no options at all when none were asked for', () async {
      router.answer(
        'POST /v1/search',
        const Answer(200, {'provider': 'exa', 'model': 'm', 'results': []}),
      );

      await agents.router().search('anything');

      expect(router.last('POST /v1/search').json, {'query': 'anything'});
    });
  });
}
