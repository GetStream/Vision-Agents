import 'dart:convert';

import 'package:flutter_assistant/settings.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  group('ReplyFormat', () {
    test('current reads the format the backend says the agent answers in', () async {
      for (final format in ReplyFormat.values) {
        final client = MockClient(
          (_) async => http.Response(jsonEncode({'format': format.name}), 200),
        );
        final read = await http.runWithClient(ReplyFormat.current, () => client);
        expect(read, format);
      }
    });

    test('choose puts the format to the backend', () async {
      String? sent;
      final client = MockClient((request) async {
        sent = '${request.method} ${request.url.path} ${request.body}';
        return http.Response(request.body, 200);
      });
      await http.runWithClient(ReplyFormat.a2ui.choose, () => client);
      expect(sent, 'PUT /reply-format ${jsonEncode({'format': 'a2ui'})}');
    });

    test('fails with what the backend said when it refuses', () async {
      final client = MockClient((_) async => http.Response('no such format', 400));
      expect(
        http.runWithClient(ReplyFormat.markdown.choose, () => client),
        throwsA(isA<Exception>()),
      );
    });
  });
}
