// The posture against a router that verifies credentials, which needs one running in `api_key`
// mode and a key minted for it with `router keys create`; the JavaScript README has the
// commands.
//
//     INTEGRATION_ROUTER_URL=http://localhost:8099 INTEGRATION_ROUTER_KEY=vak_... \
//       INTEGRATION_ROUTER_SECRET=vas_... dart test -t live test/keyed_router_posture_test.dart
@Tags(['live'])
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

final _env = Platform.environment;
final _key = _env['INTEGRATION_ROUTER_KEY'] ?? '';
final _secret = _env['INTEGRATION_ROUTER_SECRET'] ?? '';
final _url = Uri.parse(_env['INTEGRATION_ROUTER_URL'] ?? 'http://localhost:8099');

/// An HS256 token, which is what the app's own backend would have minted.
String _sign(Map<String, Object?> claims) {
  String encode(List<int> bytes) => base64Url.encode(bytes).replaceAll('=', '');
  final unsigned = [
    {'alg': 'HS256', 'typ': 'JWT'},
    claims,
  ].map((part) => encode(utf8.encode(jsonEncode(part)))).join('.');
  final signature = Hmac(sha256, utf8.encode(_secret)).convert(utf8.encode(unsigned));
  return '$unsigned.${encode(signature.bytes)}';
}

void main() {
  late VisionAgents agents;

  setUp(() {
    final id = 'dart-device-${DateTime.now().microsecondsSinceEpoch}';
    final exp = DateTime.now().add(const Duration(hours: 1)).millisecondsSinceEpoch ~/ 1000;
    final token = _sign({'user_id': id, 'exp': exp});
    agents = VisionAgents(url: _url, apiKey: _key, userId: id, token: () async => token);
  });

  tearDown(() => agents.close());

  group('a keyed router', () {
    test('refuses a device minting tokens or configuring agents', () async {
      const serverOnly = [
        ('POST', '/v1/agents/calls/default:anything/token'),
        ('POST', '/v1/agents/chat-token'),
        ('POST', '/v1/agents/guests/claim'),
        ('GET', '/v1/agents/configs'),
      ];
      for (final (method, path) in serverOnly) {
        final request = http.Request(method, agents.backend.requestUri(path))
          ..headers.addAll(await agents.backend.headers())
          ..headers['Content-Type'] = 'application/json';
        if (method == 'POST') {
          request.body = '{}';
        }
        final answered = await http.Response.fromStream(await request.send());
        final error = (jsonDecode(answered.body) as Map<String, Object?>)['error'];

        expect(answered.statusCode, 403, reason: '$method $path');
        expect((error as Map<String, Object?>)['code'], 'server_side_only');
      }
    });

    test('still lets a device list its own conversations', () async {
      await agents.sessions.query();
    });
  }, skip: _key.isEmpty ? 'INTEGRATION_ROUTER_KEY is not set' : false);
}
