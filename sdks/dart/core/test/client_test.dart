import 'dart:async';
import 'dart:convert';
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

    test('presents a token with the api key, asking the provider once', () async {
      router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
      var asked = 0;
      final keyed = VisionAgents(
        url: router.url,
        apiKey: 'key',
        token: () async => 'token-${++asked}',
      );

      for (var i = 0; i < 3; i++) {
        await keyed.sessions.get('s1');
      }
      keyed.close();

      expect(asked, 1);
      expect(router.arrived.map((request) => request.header('Authorization')), [
        'Bearer token-1',
        'Bearer token-1',
        'Bearer token-1',
      ]);
      final request = router.last('GET /v1/agents/sessions/s1');
      expect(request.query['api_key'], 'key');
      expect(request.header('X-Api-Key'), isNull);
      expect(request.header('Stream-Auth-Type'), 'jwt');
      expect(request.header('X-Customer-Id'), isNull);
    });

    group('a token provider', () {
      late int asked;
      late TokenProvider counting;
      late VisionAgents keyed;

      setUp(() {
        asked = 0;
        counting = () async => 'token-${++asked}';
        keyed = VisionAgents(url: router.url, apiKey: 'key', userId: 'jlahey', token: counting);
      });

      tearDown(() => keyed.close());

      Answer refusing(Arrived request) => request.header('Authorization') == 'Bearer token-1'
          ? Answer(401, errorJson('authentication', 'token_expired', 'token expired'))
          : Answer(200, sessionJson());

      test('is asked once by requests that need it at the same time', () async {
        router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
        final fetched = Completer<String>();
        final waiting = VisionAgents(
          url: router.url,
          apiKey: 'key',
          token: () {
            asked++;
            return fetched.future;
          },
        );

        final requests = Future.wait([for (var i = 0; i < 3; i++) waiting.sessions.get('s1')]);
        await Future<void>.delayed(Duration.zero);
        fetched.complete('token-1');
        await requests;
        waiting.close();

        expect(asked, 1);
      });

      test('is asked again after a 401, and the request sent once more', () async {
        router.answerBy('GET /v1/agents/sessions/s1', refusing);

        final session = await keyed.sessions.get('s1');

        expect(session.id, 's1');
        expect(asked, 2);
        expect(router.arrived.map((request) => request.header('Authorization')), [
          'Bearer token-1',
          'Bearer token-2',
        ]);

        // The fresh one is kept for what comes next.
        await keyed.sessions.get('s1');
        expect(asked, 2);
      });

      test('is not asked a third time when the fresh token is refused too', () async {
        router.answer(
          'POST /v1/agents/sessions',
          Answer(401, errorJson('authentication', 'invalid_token', 'not a token of this app')),
        );

        await expectLater(
          keyed.sessions.create(const SessionOptions(title: 'Plans')),
          throwsA(isA<RouterException>().having((error) => error.status, 'status', 401)),
        );
        expect(asked, 2);
        expect(router.arrived, hasLength(2));
        expect(router.arrived[1].json, isNotNull, reason: 'the body is sent again');
        expect(router.arrived[1].json, router.arrived[0].json);
      });

      test('does not retry a router that is never shown the token', () async {
        // Reached by customer id, the token is for Stream's own clients, so a fresh one would
        // send the very same request again.
        router.answer('GET /v1/agents/sessions/s1', Answer(401, errorJson('a', 'b', 'c')));
        final local = VisionAgents(
          url: router.url,
          customerId: 'acme',
          apiKey: 'key',
          userId: 'jlahey',
          token: counting,
        );

        await expectLater(local.sessions.get('s1'), throwsA(isA<RouterException>()));
        local.close();

        expect(router.arrived, hasLength(1));
        expect(asked, 0);
      });

      test('is asked again before a socket is opened with a token about to expire', () async {
        String jwt(Duration expiresIn) {
          final exp = DateTime.now().add(expiresIn).millisecondsSinceEpoch ~/ 1000;
          final claims = base64Url.encode(utf8.encode(jsonEncode({'exp': exp})));
          return 'header.${claims.replaceAll('=', '')}.signature';
        }

        final tokens = [jwt(const Duration(seconds: 10)), jwt(const Duration(hours: 1))];
        final backend = Backend(url: router.url, apiKey: 'key', token: () async => tokens[asked++]);

        await backend.headers();
        final uri = await backend.socketUri('/v1/agents/sessions/s1/events');

        expect(asked, 2, reason: 'the first expires within the margin');
        expect(uri.queryParameters['token'], tokens[1]);
        await backend.headers();
        expect(asked, 2, reason: 'the second is kept');
      });

      test(
        'is asked once more, not once per request, when requests are refused together',
        () async {
          router.answerBy('GET /v1/agents/sessions/s1', refusing);
          await keyed.backend.headers();

          final sessions = await Future.wait([
            for (var i = 0; i < 4; i++) keyed.sessions.get('s1'),
          ]);

          expect(sessions, hasLength(4));
          expect(asked, 2, reason: 'token-1, then one token-2 for every retry');
        },
      );

      test('keeps a newer token when a refusal of an older one arrives late', () async {
        await keyed.backend.headers();
        expect(keyed.backend.expireToken('token-1'), isTrue);
        expect((await keyed.backend.headers())['Authorization'], 'Bearer token-2');

        // A request sent with token-1 before the refresh is refused only now.
        expect(keyed.backend.expireToken('token-1'), isTrue, reason: 'the retry goes with token-2');
        expect((await keyed.backend.headers())['Authorization'], 'Bearer token-2');
        expect(asked, 2);
      });

      test('is asked again after a fetch that failed', () async {
        router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
        final flaky = VisionAgents(
          url: router.url,
          apiKey: 'key',
          token: () async => ++asked == 1 ? throw StateError('backend down') : 'token-$asked',
        );

        await expectLater(flaky.sessions.get('s1'), throwsStateError);
        await flaky.sessions.get('s1');
        flaky.close();

        expect(router.last('GET /v1/agents/sessions/s1').header('Authorization'), 'Bearer token-2');
      });

      test("never hands one user's token to a copy for another user", () async {
        router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
        // A provider that answers for whoever is signed in, as an app's usually does.
        var signedIn = 'alice';
        final alice = VisionAgents(
          url: router.url,
          apiKey: 'key',
          userId: 'alice',
          token: () async => 'token-for-$signedIn',
        );

        await alice.sessions.get('s1');
        signedIn = 'bob';
        final bob = alice.withUser('bob');
        await bob.sessions.get('s1');
        final credentials = await bob.backend.streamCredentials();
        alice.close();

        expect(
          router.last('GET /v1/agents/sessions/s1').header('Authorization'),
          'Bearer token-for-bob',
        );
        expect([credentials.userId, credentials.token], ['bob', 'token-for-bob']);
      });

      test(
        'is shared by a copy for the same user, and not by one with a provider of its own',
        () async {
          router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
          var otherAsked = 0;

          await keyed.sessions.get('s1');
          await keyed.withUser('jlahey').sessions.get('s1');
          await keyed.withUser('jlahey', token: counting).sessions.get('s1');
          await keyed
              .withUser('jlahey', token: () async => 'other-${++otherAsked}')
              .sessions
              .get('s1');

          expect(asked, 1);
          expect(otherAsked, 1);
          expect(router.arrived.map((request) => request.header('Authorization')), [
            'Bearer token-1',
            'Bearer token-1',
            'Bearer token-1',
            'Bearer other-1',
          ]);
        },
      );

      test(
        'keeps fetching the replacement of a token about to expire when that token is refused',
        () async {
          final claims = base64Url.encode(
            utf8.encode(jsonEncode({'exp': DateTime.now().millisecondsSinceEpoch ~/ 1000 + 10})),
          );
          final expiring = 'header.${claims.replaceAll('=', '')}.signature';
          final replacement = Completer<String>();
          final backend = Backend(
            url: router.url,
            apiKey: 'key',
            token: () => ++asked == 1 ? Future.value(expiring) : replacement.future,
          );

          await backend.headers();
          final replacing = backend.headers();
          // A request that went out with the expiring token is refused while it is replaced.
          backend.expireToken(expiring);
          replacement.complete('token-2');

          expect((await replacing)['Authorization'], 'Bearer token-2');
          expect((await backend.headers())['Authorization'], 'Bearer token-2');
          expect(asked, 2);
        },
      );

      test('drops a fetch that was overtaken by a refusal', () async {
        final fetches = [Completer<String>(), Completer<String>()];
        final backend = Backend(
          url: router.url,
          apiKey: 'key',
          token: () => fetches[asked++].future,
        );

        final stale = backend.headers();
        backend.expireToken();
        final fresh = backend.headers();
        fetches[1].complete('token-2');
        await fresh;
        fetches[0].complete('token-1');
        await stale;

        expect((await backend.headers())['Authorization'], 'Bearer token-2');
        expect(asked, 2);
      });

      test('answers at once, with no timer, when the provider does', () async {
        final backend = Backend(url: router.url, apiKey: 'key', token: () async => 'token-1');

        // A widget test's fake clock never fires a timer it is not told to.
        final headers = await runZoned(
          backend.headers,
          zoneSpecification: ZoneSpecification(
            createTimer: (self, parent, zone, duration, callback) =>
                throw StateError('the token was fetched behind a timer'),
          ),
        );

        expect(headers['Authorization'], 'Bearer token-1');
      });

      test('keeps a token whose expiry is further out than a date can hold', () async {
        router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
        final claims = base64Url.encode(utf8.encode(jsonEncode({'exp': 9999999999999})));
        final forever = VisionAgents(
          url: router.url,
          apiKey: 'key',
          token: () async => 'header.${claims.replaceAll('=', '')}.signature-${++asked}',
        );

        for (var i = 0; i < 3; i++) {
          await forever.sessions.get('s1');
        }
        forever.close();

        expect(router.arrived, hasLength(3));
        expect(asked, 1);
      });
    });

    test(
      "a key beside a customer id is Stream's, and the router is reached by customer id",
      () async {
        router.answer('GET /v1/agents/sessions/s1', Answer(200, sessionJson()));
        final local = VisionAgents(
          url: router.url,
          customerId: 'acme',
          apiKey: 'stream-key',
          userId: 'jlahey',
          token: () async => 'token-for-jlahey',
        );

        await local.sessions.get('s1');
        local.close();

        final request = router.last('GET /v1/agents/sessions/s1');
        expect(request.header('X-Customer-Id'), 'acme');
        expect(request.header('X-Stream-User-Id'), 'jlahey');
        expect(request.query['api_key'], isNull);
        expect(request.header('Authorization'), isNull);
        expect((await local.backend.streamCredentials()).apiKey, 'stream-key');
      },
    );

    test('goes to the hosted router when no url is named', () {
      final backend = Backend(apiKey: 'key', token: () async => 'token');

      expect(backend.url, Backend.defaultUrl);
      expect(backend.url.toString(), 'https://accelerate-us-east4.gcp.stream-io-api.com');
    });

    test('a region is the hosted router there', () {
      expect(
        Backend.regionUrl('us-east4').toString(),
        'https://accelerate-us-east4.gcp.stream-io-api.com',
      );
      expect(
        Backend.regionUrl('europe-west4').toString(),
        'https://accelerate-europe-west4.gcp.stream-io-api.com',
      );
    });

    test('anything but a region name is refused', () {
      for (final name in [
        '',
        'US-EAST4',
        'us-east4.evil.com/x',
        'us-east4\n',
        'us_east4',
        'useast4',
        'us-east',
      ]) {
        expect(() => Backend.regionUrl(name), throwsArgumentError, reason: name);
      }
    });

    test('an api key with no token asks for nothing but a guest', () async {
      final keyed = VisionAgents(url: router.url, apiKey: 'key');

      await expectLater(keyed.sessions.get('s1'), throwsArgumentError);
      keyed.close();

      expect(router.arrived, isEmpty);
      expect(() => Backend(url: Uri.parse('http://x')), throwsArgumentError);
    });

    test('puts credentials and a device\'s auth type in a socket url by customer id', () async {
      final backend = Backend(
        url: Uri.parse('https://router.example/base'),
        customerId: 'acme',
      ).withUser('ada');

      final url = await backend.socketUri('/v1/agents/sessions/s1/events', {'decisions': 'false'});

      expect(url.scheme, 'wss');
      expect(url.path, '/base/v1/agents/sessions/s1/events');
      expect(url.queryParameters, {
        'decisions': 'false',
        'customer_id': 'acme',
        'user_id': 'ada',
        'stream-auth-type': 'jwt',
      });
    });

    test('puts the key, the token and a device\'s auth type in a socket url by key', () async {
      for (final url in [
        Uri.parse('http://localhost:8080'),
        Backend.defaultUrl,
        Backend.regionUrl('europe-west4'),
      ]) {
        final backend = Backend(url: url, apiKey: 'key', token: () async => 'token');

        final socket = await backend.socketUri('/v1/agents/sessions/s1/events');

        expect(socket.queryParameters, {
          'api_key': 'key',
          'token': 'token',
          'authorization': 'token',
          'stream-auth-type': 'jwt',
        });
      }
    });
  });

  group('Sessions', () {
    test('opens a written conversation, sending only what the caller set', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      final session = await agents.sessions.create(
        const SessionOptions(
          id: 'order-1042',
          agent: 'docs',
          greeting: Greeting('Hello.', mode: GreetingMode.variation),
          title: 'Billing',
          projectId: 'Health',
          modelOverwrites: ModelOverwrites(thinking: Thinking.high),
          returnUrl: 'com.example.app:/connected',
        ),
      );

      expect(router.last('POST /v1/agents/sessions').json, {
        'id': 'order-1042',
        'agent': 'docs',
        'greeting': {'text': 'Hello.', 'mode': 'variation'},
        'title': 'Billing',
        'project_id': 'Health',
        'model_overwrites': {'thinking': 'high'},
        'return_url': 'com.example.app:/connected',
      });
      expect(session.id, 's1');
      expect(session.isText, isTrue);
      expect(session.mode, SessionMode.text);
      expect(session.modality, SessionModality.text);
      expect(session.createdAt, DateTime.utc(2026, 9, 24, 15, 54, 56, 38, 55));
    });

    test('an empty return url sends none, so the sign-in goes back to the dashboard', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.sessions.create(const SessionOptions(returnUrl: ''));

      expect(router.last('POST /v1/agents/sessions').json, <String, Object?>{});
    });

    test("memory is the signed-in user's, narrowed by the labels asked for", () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));
      final ada = agents.withUser('ada');

      await ada.sessions.create(
        const SessionOptions(
          memory: SessionMemory(filter: {'topic': 'billing'}, appId: 'web'),
          timezone: 'Europe/Amsterdam',
        ),
      );
      final narrowed = router.last('POST /v1/agents/sessions').json;
      await ada
          .agent('docs')
          .sessions
          .create(const SessionOptions(memory: SessionMemory(), incognito: true));

      expect(narrowed, {
        'memory': {
          'app_id': 'web',
          'filter': {'topic': 'billing'},
          'user_id': 'ada',
        },
        'timezone': 'Europe/Amsterdam',
      });
      // The router keeps nothing of an incognito session, so the two are sent as asked and it
      // decides.
      expect(router.last('POST /v1/agents/sessions').json, {
        'agent': 'docs',
        'incognito': true,
        'memory': {'user_id': 'ada'},
      });
    });

    test('remembering with nobody named sends no user, and no memory is none at all', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.sessions.create(const SessionOptions(memory: SessionMemory()));
      final remembering = router.last('POST /v1/agents/sessions').json;
      await agents.withUser('ada').sessions.create(const SessionOptions(timezone: ''));

      expect(remembering, {'memory': <String, Object?>{}});
      expect(router.last('POST /v1/agents/sessions').json, <String, Object?>{});
    });

    test('a voice session asks to start voice and is on its own call', () async {
      router.answer(
        'POST /v1/agents/sessions',
        Answer(201, {...sessionJson(), 'call_id': 's1', 'text': false}),
      );

      final session = await agents.sessions.create(
        const SessionOptions(configId: 'c1', startVoice: true),
      );

      expect(router.last('POST /v1/agents/sessions').json, {
        'config_id': 'c1',
        'start_voice': true,
      });
      expect(session.callId, 's1');
      expect(session.callType, 'agent');
      expect(session.isText, isFalse);
    });

    test('starting and stopping voice moves a session onto its call and off it', () async {
      router
        ..answer(
          'POST /v1/agents/sessions/s1/voice',
          Answer(200, {...sessionJson(), 'call_id': 's1', 'text': false}),
        )
        ..answer('DELETE /v1/agents/sessions/s1/voice', Answer(200, sessionJson()));

      final started = await agents.sessions.startVoice('s1');
      final stopped = await agents.sessions.stopVoice('s1');

      expect(router.arrived.map((request) => '${request.method} ${request.path}'), [
        'POST /v1/agents/sessions/s1/voice',
        'DELETE /v1/agents/sessions/s1/voice',
      ]);
      expect(started.callId, 's1');
      expect(stopped.callId, isEmpty);
      expect(stopped.isText, isTrue);
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

    test('naming a config by id does not also send the agent it is scoped to', () async {
      router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));

      await agents.agent('docs').sessions.create(const SessionOptions(configId: 'c2'));

      expect(router.last('POST /v1/agents/sessions').json, {'config_id': 'c2'});
    });

    group('querying sessions', () {
      const route = 'POST /v1/agents/sessions/query';

      setUp(() => router.answer(route, Answer(200, pageJson([]))));

      Object? sent() => router.last(route).json;

      test("an agent's sessions are opened against it and queried by it alone", () async {
        router.answer('POST /v1/agents/sessions', Answer(201, sessionJson()));
        final docs = agents.agent('docs');

        await docs.sessions.create();
        await docs.sessions.query();

        expect(router.last('POST /v1/agents/sessions').json, {'agent': 'docs'});
        expect(sent(), {
          'filter': {'agent': 'docs'},
        });
      });

      test('a query with nothing to narrow it sends no filter', () async {
        final page = await agents.sessions.search('');

        expect(sent(), <String, Object?>{});
        expect(page.items, isEmpty);
        expect(page.nextCursor, isNull);
      });

      test('narrows to a project', () async {
        await agents.agent('docs').sessions.query(filter: {'project_id': 'a'});

        expect(sent(), {
          'filter': {'agent': 'docs', 'project_id': 'a'},
        });
      });

      test(
        'narrows to either of two projects, the agent beside the \$or and not around it',
        () async {
          await agents
              .agent('docs')
              .sessions
              .query(
                filter: {
                  r'$or': [
                    {'project_id': 'a'},
                    {'project_id': 'b'},
                  ],
                },
                sort: const [
                  SessionSort(field: 'pinned'),
                  SessionSort(field: 'last_message_at'),
                ],
                limit: 20,
                cursor: null,
              );

          expect(sent(), {
            'filter': {
              r'$or': [
                {'project_id': 'a'},
                {'project_id': 'b'},
              ],
              'agent': 'docs',
            },
            'sort': [
              {'direction': -1, 'field': 'pinned'},
              {'direction': -1, 'field': 'last_message_at'},
            ],
            'limit': 20,
          });
        },
      );

      test('sends a nested \$and and \$or as written', () async {
        await agents
            .agent('docs')
            .sessions
            .query(
              filter: {
                r'$and': [
                  {
                    'text': {r'$q': 'refund'},
                  },
                  {
                    r'$or': [
                      {
                        'project_id': {r'$eq': 'a'},
                      },
                      {
                        r'$and': [
                          {'project_id': 'b'},
                          {
                            'agent': {r'$eq': 'docs'},
                          },
                        ],
                      },
                    ],
                  },
                ],
              },
            );

        expect(sent(), {
          'filter': {
            r'$and': [
              {
                'text': {r'$q': 'refund'},
              },
              {
                r'$or': [
                  {
                    'project_id': {r'$eq': 'a'},
                  },
                  {
                    r'$and': [
                      {'project_id': 'b'},
                      {
                        'agent': {r'$eq': 'docs'},
                      },
                    ],
                  },
                ],
              },
            ],
            'agent': 'docs',
          },
        });
      });

      test('narrows to any of several projects with \$in', () async {
        await agents.sessions.query(
          filter: {
            'project_id': {
              r'$in': ['a', 'b', 'c'],
            },
          },
        );

        expect(sent(), {
          'filter': {
            'project_id': {
              r'$in': ['a', 'b', 'c'],
            },
          },
        });
      });

      test("an agent's own sessions stay that agent's, whatever the filter names", () async {
        await agents.agent('docs').sessions.query(filter: {'agent': 'billing', 'project_id': 'a'});

        expect(sent(), {
          'filter': {'agent': 'docs', 'project_id': 'a'},
        });
      });

      test('searches by the words a conversation was titled with, within a project', () async {
        router.answer(route, Answer(200, pageJson([sessionJson(id: 's2')])));

        final found = await agents
            .agent('docs')
            .sessions
            .search("o'brien billing", filter: {'project_id': 'a'}, limit: 5, cursor: 'c1');

        expect(found.items.single.id, 's2');
        expect(sent(), {
          'filter': {
            'agent': 'docs',
            'project_id': 'a',
            'text': {r'$q': "o'brien billing"},
          },
          'limit': 5,
          'cursor': 'c1',
        });
      });

      test('a search refuses a filter that already has text', () async {
        expect(
          () => agents.sessions.search(
            'refund',
            filter: {
              'text': {r'$q': 'billing'},
            },
          ),
          throwsArgumentError,
        );
        expect(
          () => agents.sessions.search(
            'refund',
            filter: {
              r'$and': [
                {
                  'text': {r'$q': 'billing'},
                },
              ],
            },
          ),
          throwsArgumentError,
        );
        expect(router.arrived, isEmpty);
      });

      test('sends each order a device may ask for', () async {
        final sorts = <List<SessionSort>, List<Map<String, Object?>>>{
          const [SessionSort(field: 'last_message_at')]: [
            {'direction': -1, 'field': 'last_message_at'},
          ],
          const [SessionSort(field: 'last_message_at', direction: 1)]: [
            {'direction': 1, 'field': 'last_message_at'},
          ],
          const [SessionSort(field: 'pinned')]: [
            {'direction': -1, 'field': 'pinned'},
          ],
          const [
            SessionSort(field: 'pinned'),
            SessionSort(field: 'last_message_at', direction: 1),
          ]: [
            {'direction': -1, 'field': 'pinned'},
            {'direction': 1, 'field': 'last_message_at'},
          ],
        };

        for (final MapEntry(key: sort, value: wire) in sorts.entries) {
          await agents.sessions.query(sort: sort);
          expect(sent(), {'sort': wire});
        }

        await agents.sessions.query(
          filter: {
            'text': {r'$q': 'refund'},
          },
          sort: const [SessionSort(field: 'relevance')],
        );
        expect(sent(), {
          'filter': {
            'text': {r'$q': 'refund'},
          },
          'sort': [
            {'direction': -1, 'field': 'relevance'},
          ],
        });
      });

      test('pages with the cursor of the page before', () async {
        router.answerInTurn(route, [
          Answer(200, pageJson([sessionJson()], nextCursor: 'c2')),
          Answer(200, pageJson([sessionJson(id: 's2')])),
        ]);
        const sort = [SessionSort(field: 'pinned')];

        final first = await agents.sessions.query(sort: sort, limit: 1);
        final second = await agents.sessions.query(sort: sort, limit: 1, cursor: first.nextCursor);

        expect(first.items.single.id, 's1');
        expect(first.hasMore, isTrue);
        expect(first.nextCursor, 'c2');
        expect(sent(), {
          'sort': [
            {'direction': -1, 'field': 'pinned'},
          ],
          'limit': 1,
          'cursor': 'c2',
        });
        expect(second.items.single.id, 's2');
        expect(second.hasMore, isFalse);
        expect(second.nextCursor, isNull);
      });

      test('refuses what only a backend may filter or order by, before sending anything', () async {
        final refused = <Map<String, Object?>>[
          {'review_score': 5},
          {'user_id': 'jlahey'},
          {
            r'$and': [
              {
                r'$or': [
                  {'project_id': 'a'},
                  {
                    'review_score': {r'$gte': 4},
                  },
                ],
              },
            ],
          },
          {
            r'$and': [
              {'user_id': 'jlahey'},
            ],
          },
        ];

        for (final filter in refused) {
          expect(() => agents.sessions.query(filter: filter), throwsArgumentError);
          expect(() => agents.sessions.search('refund', filter: filter), throwsArgumentError);
        }
        expect(
          () => agents.sessions.query(sort: const [SessionSort(field: 'review_score')]),
          throwsArgumentError,
        );
        expect(router.arrived, isEmpty);
      });

      test('refuses a field sessions are no longer filtered on, rather than dropping it', () async {
        for (final field in ['state', 'modality', 'agent_id', 'config_id', 'team', 'created_at']) {
          expect(() => agents.sessions.query(filter: {field: 'x'}), throwsArgumentError);
        }
        expect(
          () => agents.sessions.query(
            filter: {
              r'$or': {'project_id': 'a'},
            },
          ),
          throwsArgumentError,
        );
        expect(() => agents.sessions.query(filter: {'text': 'refund'}), throwsArgumentError);
        expect(router.arrived, isEmpty);
      });
    });

    test('pins and unpins a conversation, and reads when it was pinned', () async {
      router.answerInTurn('PATCH /v1/agents/sessions/s1', [
        Answer(200, {...sessionJson(), 'pinned_at': '2026-11-11T10:00:00.5Z'}),
        Answer(200, sessionJson()),
        Answer(200, {...sessionJson(), 'title': 'Billing', 'pinned_at': '2026-11-11T10:00:01Z'}),
      ]);

      final pinned = await agents.sessions.pin('s1');
      final pin = router.last('PATCH /v1/agents/sessions/s1').json;
      final unpinned = await agents.sessions.unpin('s1');
      final unpin = router.last('PATCH /v1/agents/sessions/s1').json;
      final both = await agents.sessions.update('s1', title: 'Billing', pinned: true);

      expect(pin, {'pinned': true});
      expect(pinned.pinnedAt, DateTime.utc(2026, 11, 11, 10, 0, 0, 500));
      expect(unpin, {'pinned': false});
      expect(unpinned.pinnedAt, isNull);
      expect(router.last('PATCH /v1/agents/sessions/s1').json, {
        'title': 'Billing',
        'pinned': true,
      });
      expect(both.pinnedAt, DateTime.utc(2026, 11, 11, 10, 0, 1));
    });

    test('reads which of a page of conversations are pinned', () async {
      router.answer(
        'POST /v1/agents/sessions/query',
        Answer(
          200,
          pageJson([
            {...sessionJson(), 'pinned_at': '2026-11-11T10:00:00Z'},
            sessionJson(id: 's2'),
          ]),
        ),
      );

      final page = await agents.sessions.query(sort: const [SessionSort(field: 'pinned')]);

      expect(page.items.first.pinnedAt, DateTime.utc(2026, 11, 11, 10));
      expect(page.items.last.pinnedAt, isNull);
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
        'request_id': matches(RegExp(r'^[0-9a-f]{32}$')),
        'text': 'What is my name?',
        'images': [
          {'url': 'https://x/cat.png', 'detail': 'low'},
        ],
      });
    });

    test('names a text question with a request id it generates', () async {
      router.answer(
        'POST /v1/agents/sessions/s1/responses',
        Answer(202, responseJson('r1', status: 'running')),
      );

      await agents.sessions.responses('s1').create('Hello');

      expect(router.last('POST /v1/agents/sessions/s1/responses').json, {
        'request_id': matches(RegExp(r'^[0-9a-f]{32}$')),
        'text': 'Hello',
      });
    });

    test('caps one answer and sets how long it thinks, still as a text question', () async {
      router.answer(
        'POST /v1/agents/sessions/s1/responses',
        Answer(202, responseJson('r1', status: 'running')),
      );

      await agents.sessions
          .responses('s1')
          .create('Think hard', maxOutputTokens: 512, reasoningEffort: 'high');

      expect(router.last('POST /v1/agents/sessions/s1/responses').json, {
        'max_output_tokens': 512,
        'reasoning_effort': 'high',
        'request_id': matches(RegExp(r'^[0-9a-f]{32}$')),
        'text': 'Think hard',
      });
    });

    test(
      'sends what the person did in a reply\'s UI with their words, still under a request id',
      () async {
        router.answer(
          'POST /v1/agents/sessions/s1/responses',
          Answer(202, responseJson('r1', status: 'running')),
        );
        final responses = agents.sessions.responses('s1');

        await responses.create(
          'Update the plan',
          uiAction: {
            'v': 1,
            'message_id': 'reply-1',
            'format': 'a2ui',
            'name': 'update_plan',
            'payload': {
              'pace': 'relaxed',
              'nights': 2,
              'interests': ['food', 'museums'],
              'flexible': true,
            },
          },
        );
        final acted = router.last('POST /v1/agents/sessions/s1/responses').json;
        await responses.create('Plain', uiAction: {});
        final empty = router.last('POST /v1/agents/sessions/s1/responses').json;

        expect(acted, {
          'request_id': matches(RegExp(r'^[0-9a-f]{32}$')),
          'text': 'Update the plan',
          'ui_action': {
            'v': 1,
            'message_id': 'reply-1',
            'format': 'a2ui',
            'name': 'update_plan',
            'payload': {
              'pace': 'relaxed',
              'nights': 2,
              'interests': ['food', 'museums'],
              'flexible': true,
            },
          },
        });
        expect(empty, {
          'request_id': matches(RegExp(r'^[0-9a-f]{32}$')),
          'text': 'Plain',
        }, reason: 'an empty action is not sent');
      },
    );

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

    test('remembers the refresh secret a minted guest is handed', () async {
      router.answer(
        'POST /v1/agents/guests',
        Answer(201, {...guest('guest-1'), 'refresh_secret': 's1'}),
      );
      final store = MemoryGuestStore();

      await agents.guestUser(store: store);

      expect((jsonDecode((await store.read())!) as Map)['refresh_secret'], 's1');
    });

    test('a guest whose token expired comes back with their refresh secret', () async {
      router.answer(
        'POST /v1/agents/guests',
        Answer(201, {...guest('guest-1', token: 't2'), 'refresh_secret': 's2'}),
      );
      final store = MemoryGuestStore();
      await store.write(
        '{"id":"guest-1","token":"t1","refresh_secret":"s1","expires_at":"${DateTime.now().subtract(const Duration(hours: 1)).toUtc().toIso8601String()}"}',
      );

      final renewed = await agents.guestUser(store: store);

      expect(renewed.token, 't2');
      expect(router.arrived.single.json, {'id': 'guest-1', 'refresh_secret': 's1'});
      expect((jsonDecode((await store.read())!) as Map)['refresh_secret'], 's2');
    });

    test('a spent secret is answered with somebody new, who is remembered instead', () async {
      router.answer(
        'POST /v1/agents/guests',
        Answer(201, {...guest('guest-2', token: 't2'), 'refresh_secret': 's9'}),
      );
      final store = MemoryGuestStore();
      await store.write(
        '{"id":"guest-1","token":"t1","refresh_secret":"spent","expires_at":"${DateTime.now().subtract(const Duration(hours: 1)).toUtc().toIso8601String()}"}',
      );

      final other = await agents.guestUser(store: store);
      final again = await agents.guestUser(store: store);

      expect(other.id, 'guest-2');
      expect(again, other);
      expect(router.arrived, hasLength(1));
      final stored = jsonDecode((await store.read())!) as Map;
      expect(stored['id'], 'guest-2');
      expect(stored['refresh_secret'], 's9');
    });

    test('an expired guest with no secret does not name an id it cannot prove', () async {
      router.answer('POST /v1/agents/guests', Answer(201, guest('guest-2')));
      final store = MemoryGuestStore();
      await store.write(
        '{"id":"guest-1","token":"t1","expires_at":"${DateTime.now().subtract(const Duration(hours: 1)).toUtc().toIso8601String()}"}',
      );

      final other = await agents.guestUser(name: 'Guest', store: store);

      expect(other.id, 'guest-2');
      expect(router.arrived.single.json, {'name': 'Guest'});
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

    test('a device with only the app key asks for a guest anonymously', () async {
      router.answer('POST /v1/agents/guests', Answer(201, guest('guest-1')));
      final keyed = VisionAgents(url: router.url, apiKey: 'key');

      final minted = await keyed.guestUser();
      keyed.close();

      expect(minted.id, 'guest-1');
      final request = router.last('POST /v1/agents/guests');
      expect(request.header('Stream-Auth-Type'), 'anonymous');
      expect(request.header('Authorization'), isNull);
      expect(request.query['api_key'], 'key');
    });

    test('a router admitting no anonymous callers says to mint the guest on the backend', () async {
      router.answer(
        'POST /v1/agents/guests',
        const Answer(401, {
          'code': 2,
          'message': 'anonymous access is not available on the acceleration api',
          'StatusCode': 401,
        }),
      );
      final keyed = VisionAgents(url: router.url, apiKey: 'key');

      await expectLater(
        keyed.guestUser(),
        throwsA(isA<StateError>().having((e) => e.message, 'message', contains('withGuest'))),
      );
      keyed.close();
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
