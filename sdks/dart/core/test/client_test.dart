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

    test('goes to the hosted router when no url is named', () {
      final backend = Backend(apiKey: 'key', token: () async => 'token');

      expect(backend.url, Backend.defaultUrl);
      expect(backend.url.toString(), 'https://accelerate.gcp.stream-io-api.com');
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
          id: '0199a3f2-7c1e-7d4a-9b2e-5f6a7b8c9d0e',
          agent: 'docs',
          title: 'Billing',
          projectId: 'Health',
          modelOverwrites: ModelOverwrites(thinking: Thinking.high),
        ),
      );

      expect(router.last('POST /v1/agents/sessions').json, {
        'id': '0199a3f2-7c1e-7d4a-9b2e-5f6a7b8c9d0e',
        'text': true,
        'agent': 'docs',
        'title': 'Billing',
        'project_id': 'Health',
        'model_overwrites': {'thinking': 'high'},
      });
      expect(session.id, 's1');
      expect(session.isText, isTrue);
      expect(session.mode, SessionMode.text);
      expect(session.modality, SessionModality.text);
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
              displayTitle: 'Looking up your order',
              executor: ToolExecutor.client,
              run: (_) async => '',
            ),
          ],
        ),
      );

      expect((router.last('POST /v1/agents/sessions').json as Map)['tools'], [
        {
          'name': 'lookup_order',
          'description': 'Look up an order.',
          'display_title': 'Looking up your order',
          'executor': 'client',
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

    test("an agent's sessions are opened against it and queried by it, a page at a time", () async {
      router
        ..answer('POST /v1/agents/sessions', Answer(201, sessionJson()))
        ..answer(
          'POST /v1/agents/sessions/query',
          Answer(200, pageJson([sessionJson()], nextCursor: 'c2')),
        );
      final docs = agents.agent('docs');

      await docs.sessions.create();
      final page = await docs.sessions.query(
        const SessionQuery(
          agentId: 'a1',
          modality: SessionModality.voice,
          state: SessionState.ended,
          limit: 10,
          cursor: 'c1',
        ),
      );

      expect(router.last('POST /v1/agents/sessions').json, {'text': true, 'agent': 'docs'});
      expect(router.last('POST /v1/agents/sessions/query').json, {
        'filter': {'agent': 'docs', 'agent_id': 'a1', 'modality': 'voice', 'state': 'ended'},
        'limit': 10,
        'cursor': 'c1',
      });
      expect(page.items.single.id, 's1');
      expect(page.hasMore, isTrue);
      expect(page.nextCursor, 'c2');
    });

    test('a query with nothing to narrow it sends no filter', () async {
      router.answer('POST /v1/agents/sessions/query', Answer(200, pageJson([])));

      final page = await agents.sessions.search('');

      expect(router.last('POST /v1/agents/sessions/query').json, <String, Object?>{});
      expect(page.items, isEmpty);
      expect(page.nextCursor, isNull);
    });

    test('naming a config by id does not also send the agent it is scoped to', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.agent('docs').sessions.create(const SessionOptions(configId: 'c2'));

      expect(router.last('POST /v1/agents/sessions').json, {'text': true, 'config_id': 'c2'});
    });

    test('searches by the words a conversation was titled with', () async {
      router.answer(
        'POST /v1/agents/sessions/query',
        Answer(200, pageJson([sessionJson(id: 's2')])),
      );

      final found = await agents.sessions.search(
        "o'brien billing",
        const SessionQuery(state: SessionState.live),
      );

      expect(found.items.single.id, 's2');
      expect(router.last('POST /v1/agents/sessions/query').json, {
        'filter': {
          'state': 'live',
          'text': {r'$q': "o'brien billing"},
        },
      });
    });

    test('stopping keeps the conversation and deleting takes it away', () async {
      router
        ..answer('POST /v1/agents/sessions/s1/stop', const Answer(204))
        ..answer('DELETE /v1/agents/sessions/s1', const Answer(204));

      await agents.sessions.stop('s1');
      await agents.sessions.delete('s1');

      expect(router.arrived.map((request) => '${request.method} ${request.path}'), [
        'POST /v1/agents/sessions/s1/stop',
        'DELETE /v1/agents/sessions/s1',
      ]);
    });

    test('renames a conversation, sending only what changes, and reads back the result', () async {
      router.answer(
        'PATCH /v1/agents/sessions/s1',
        Answer(200, {...sessionJson(state: 'ended'), 'title': 'Billing'}),
      );

      final session = await agents.sessions.update('s1', title: 'Billing', custom: {});

      expect(router.last('PATCH /v1/agents/sessions/s1').json, {
        'title': 'Billing',
        'custom': <String, Object?>{},
      });
      expect(session.title, 'Billing');
      expect(session.state, SessionState.ended);
    });

    test('encodes a session id into the path rather than trusting it', () async {
      router.answer('DELETE /v1/agents/sessions/a%2Fb', const Answer(204));

      await agents.sessions.delete('a/b');

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
      router.answer(
        'GET /v1/agents/sessions/s1',
        Answer(404, errorJson('not_found', 'session_not_found', 'no such session')),
      );

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

    test('reads every field of the envelope, and the request id to quote', () async {
      router.answer(
        'GET /v1/agents/sessions/s1',
        Answer(500, errorJson('internal', 'internal_error', 'something went wrong'), {
          'X-Request-Id': 'req-7f3a',
        }),
      );

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(
          isA<RouterException>()
              .having((e) => e.status, 'status', 500)
              .having((e) => e.type, 'type', RouterErrorType.internal)
              .having((e) => e.code, 'code', 'internal_error')
              .having((e) => e.message, 'message', 'something went wrong')
              .having(
                (e) => e.docUrl,
                'docUrl',
                Uri.parse('https://getstream.io/agents/docs/api/errors/#internal_error'),
              )
              .having((e) => e.requestId, 'requestId', 'req-7f3a'),
        ),
      );
    });

    test('a type and a code newer than this SDK are read, not refused', () async {
      router.answer(
        'GET /v1/agents/sessions/s1',
        Answer(418, errorJson('teapot', 'brewing', 'short and stout')),
      );

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(
          isA<RouterException>()
              .having((e) => e.type, 'type', RouterErrorType.unknown)
              .having((e) => e.code, 'code', 'brewing')
              .having((e) => e.message, 'message', 'short and stout'),
        ),
      );
    });

    test('a 403 is a server-side only path, and says so', () async {
      router.answer(
        'GET /v1/agents/sessions/s1',
        Answer(403, errorJson('permission', 'server_side_only', 'server side only')),
      );

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(
          isA<RouterException>()
              .having((e) => e.isServerSideOnly, 'isServerSideOnly', true)
              .having((e) => e.type, 'type', RouterErrorType.permission)
              .having((e) => e.code, 'code', 'server_side_only'),
        ),
      );
    });

    test('keeps what a proxy said when the refusal is not the envelope', () async {
      router.answer(
        'GET /v1/agents/sessions/s1',
        const Answer(502, '<html>bad gateway</html>', {'X-Request-Id': 'req-edge'}),
      );

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(
          isA<RouterException>()
              .having((e) => e.status, 'status', 502)
              .having((e) => e.message, 'message', '<html>bad gateway</html>')
              .having((e) => e.type, 'type', isNull)
              .having((e) => e.code, 'code', isNull)
              .having((e) => e.docUrl, 'docUrl', isNull)
              .having((e) => e.requestId, 'requestId', 'req-edge'),
        ),
      );
    });

    test("an older router's error string is kept as it came, with no code", () async {
      router.answer('GET /v1/agents/sessions/s1', const Answer(404, {'error': 'no such session'}));

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(
          isA<RouterException>()
              .having((e) => e.message, 'message', '{"error":"no such session"}')
              .having((e) => e.type, 'type', isNull)
              .having((e) => e.code, 'code', isNull)
              .having((e) => e.requestId, 'requestId', isNull),
        ),
      );
    });

    test('an empty refusal says its status phrase, and a long one is cut short', () async {
      router.answerInTurn('GET /v1/agents/sessions/s1', [
        const Answer(503),
        Answer(502, 'x' * 5000),
      ]);

      await expectLater(
        agents.sessions.get('s1'),
        throwsA(isA<RouterException>().having((e) => e.message, 'message', 'Service Unavailable')),
      );
      await expectLater(
        agents.sessions.get('s1'),
        throwsA(isA<RouterException>().having((e) => e.message.length, 'length', 1001)),
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

    test('names a question with the command id it was given', () async {
      router.answer(
        'POST /v1/agents/sessions/s1/responses',
        Answer(202, responseJson('r1', status: 'running')),
      );

      await agents.sessions.responses('s1').create('Hello', commandId: 'cmd-1');

      expect(router.last('POST /v1/agents/sessions/s1/responses').json, {
        'command_id': 'cmd-1',
        'text': 'Hello',
      });
    });

    test('lists a page of turns, oldest first, with their finish times', () async {
      router.answer(
        'GET /v1/agents/sessions/s1/responses',
        Answer(
          200,
          pageJson([responseJson('r1'), responseJson('r2', status: 'cancelled')], nextCursor: 'n'),
        ),
      );

      final turns = await agents.sessions.responses('s1').list(limit: 5, cursor: 'c');

      expect(turns.items.map((turn) => turn.id), ['r1', 'r2']);
      expect(turns.items.last.status, ResponseStatus.cancelled);
      expect(turns.items.first.finishedAt, DateTime.utc(2026, 9, 24, 15, 54, 57, 500));
      expect(turns.nextCursor, 'n');
      expect(router.last('GET /v1/agents/sessions/s1/responses').query, {
        'limit': '5',
        'cursor': 'c',
      });
    });

    test('unwinds every item a page at a time, following the cursor to the last', () async {
      Map<String, Object?> item(int ordinal) => {
        'response_id': 'r1',
        'ordinal': ordinal,
        'kind': ordinal == 0 ? 'said' : 'tool_call',
        'tool_name': 'lookup_order',
        'at': '2026-09-24T15:54:56Z',
      };
      router.answerInTurn('GET /v1/agents/sessions/s1/responses/items', [
        Answer(200, pageJson([item(0), item(1)], nextCursor: 'c2')),
        Answer(200, pageJson([item(2)])),
      ]);

      final items = await agents.sessions
          .responses('s1')
          .unwind(responseId: 'r1', pageSize: 2)
          .toList();

      expect(items.map((item) => item.kind), [ItemKind.said, ItemKind.toolCall, ItemKind.toolCall]);
      expect(router.arrived.map((request) => request.query), [
        {'response_id': 'r1', 'limit': '2'},
        {'response_id': 'r1', 'limit': '2', 'cursor': 'c2'},
      ]);
    });

    test('rewinds to a response, which answers nothing', () async {
      router.answer('POST /v1/agents/sessions/s1/rewind', const Answer(204));

      await agents.sessions.responses('s1').rewind('r1');

      expect(router.last('POST /v1/agents/sessions/s1/rewind').json, {'response_id': 'r1'});
    });

    test('a conversation kept in chat cannot be rewound, and the router says to fork it', () async {
      router.answer(
        'POST /v1/agents/sessions/s1/rewind',
        Answer(
          400,
          errorJson(
            'invalid_request',
            'invalid_request',
            'a persistent conversation cannot be rewound; fork it at the response',
          ),
        ),
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
        ..answer('POST /v1/agents/sessions/query', Answer(200, pageJson([])));

      final minted = await agents.guestUser();
      await agents.withGuest(minted).sessions.query();

      expect(router.last('POST /v1/agents/sessions/query').header('X-Stream-User-Id'), 'guest-1');
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
