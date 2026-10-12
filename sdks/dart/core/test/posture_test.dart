import 'dart:io';

import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';
import 'package:yaml/yaml.dart';

import 'support/fixtures.dart';
import 'support/router.dart';

/// Every operation the spec marks `x-client-accessible: true`, as a method and a path pattern.
List<(String, RegExp)> _clientAccessible() {
  final spec =
      loadYaml(File('../../../acceleration/api/openapi.yaml').readAsStringSync()) as YamlMap;
  return [
    for (final MapEntry(key: path as String, value: methods as YamlMap)
        in (spec['paths'] as YamlMap).entries)
      for (final MapEntry(key: method as String, value: operation) in methods.entries)
        if (operation is YamlMap && operation['x-client-accessible'] == true)
          (method.toUpperCase(), RegExp('^${path.replaceAll(RegExp(r'\{[^}]+\}'), '[^/]+')}\$')),
  ];
}

/// What a device holding this SDK can reach: a conversation, and nothing that is the app's
/// backend's to do.
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

  test('everything the SDK can ask for is something the spec lets a device ask for', () async {
    const s = '/v1/agents/sessions/s1';
    router
      ..answer('POST /v1/agents/sessions', Answer(201, sessionJson()))
      ..answer('POST /v1/agents/sessions/query', Answer(200, pageJson([])))
      ..answer('GET $s', Answer(200, sessionJson()))
      ..answer('PATCH $s', Answer(200, sessionJson()))
      ..answer('POST $s/voice', Answer(200, sessionJson()))
      ..answer('DELETE $s/voice', Answer(200, sessionJson()))
      ..answer('POST $s/fork', Answer(201, sessionJson(id: 's2')))
      ..answer('POST $s/responses', Answer(202, responseJson('r1')))
      ..answer('GET $s/responses', Answer(200, pageJson([])))
      ..answer('GET $s/responses/items', Answer(200, pageJson([])))
      ..answer('POST $s/rewind', const Answer(204))
      ..answer('POST $s/stop', const Answer(204))
      ..answer('DELETE $s', const Answer(204))
      ..answer(
        'POST /v1/search',
        const Answer(200, {
          'provider': 'exa',
          'model': 'exa-fast',
          'answer': 'Yes.',
          'results': [],
        }),
      )
      ..answer('POST /v1/agents/guests', const Answer(201, {'id': 'guest-1', 'token': 't'}));
    final sessions = agents.sessions;

    await sessions.create();
    await sessions.query();
    await sessions.search('billing');
    await sessions.get('s1');
    await sessions.update('s1', title: 'Billing');
    await sessions.startVoice('s1');
    await sessions.stopVoice('s1');
    await sessions.fork('s1');
    await sessions.responses('s1').create('hi');
    await sessions.responses('s1').list();
    await sessions.responses('s1').items();
    await sessions.responses('s1').rewind('r1');
    await sessions.stop('s1');
    await sessions.delete('s1');
    await agents.router().search('pricing');
    await agents.guestUser();

    final accessible = _clientAccessible();
    bool allowed(String method, String path) =>
        accessible.any((operation) => operation.$1 == method && operation.$2.hasMatch(path));
    expect(allowed('POST', '/v1/agents/sessions'), isTrue);
    expect(allowed('POST', '/v1/agents/calls/x/token'), isFalse);
    expect(router.arrived, hasLength(16));
    expect(
      [
        for (final request in router.arrived)
          if (!allowed(request.method, request.path)) '${request.method} ${request.path}',
      ],
      isEmpty,
      reason: 'server-side only',
    );
    expect(router.arrived.every((request) => request.header('Stream-Auth-Type') == 'jwt'), isTrue);
  });
}
