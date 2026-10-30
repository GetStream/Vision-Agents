import 'dart:convert';

import 'package:http/http.dart' as http;

import 'assistant.dart';
import 'device_tools.dart';

/// Sending from the person's Gmail.
///
/// The person connects Gmail in a chat, the first time the agent reaches for it, and the
/// router keeps the Google credential, which the app never sees. Google's Gmail MCP server
/// writes drafts but cannot send, so a send goes through the app's backend, which posts to
/// Gmail's REST API through the router's proxy for the same connection.
abstract final class Gmail {
  /// Sends a plain-text email from the person's Gmail, through the backend, and answers with
  /// the sent message's id.
  ///
  /// Throws a [ToolFailure] with the backend's reason when it could not send, such as when
  /// Gmail is not connected yet.
  static Future<String> send({
    required String to,
    required String subject,
    required String body,
  }) async {
    final response = await http.post(
      Assistant.backendUrl.resolve('/gmail/send'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'user_id': Assistant.userId, 'to': to, 'subject': subject, 'body': body}),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final said = response.body.trim();
      throw ToolFailure(said.isEmpty ? 'The backend could not reach Gmail.' : said);
    }
    return (jsonDecode(response.body) as Map<String, Object?>)['id']! as String;
  }
}
