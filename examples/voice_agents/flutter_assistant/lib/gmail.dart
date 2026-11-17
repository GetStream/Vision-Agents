import 'dart:async';
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
  /// How long the backend has to send it.
  static const _timeout = Duration(seconds: 30);

  /// Sends a plain-text email from the person's Gmail, through the backend.
  ///
  /// Throws a [ToolFailure] with the backend's reason when it could not send, such as when
  /// Gmail is not connected yet. Once the backend says it sent, nothing more is read from it:
  /// a failure after that would have the model offer to send it again.
  static Future<void> send({
    required String to,
    required String subject,
    required String body,
  }) async {
    final http.Response response;
    try {
      response = await http
          .post(
            Assistant.backendUrl.resolve('/gmail/send'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'user_id': Assistant.userId,
              'to': to,
              'subject': subject,
              'body': body,
            }),
          )
          .timeout(_timeout);
    } on http.ClientException {
      throw const ToolFailure('The phone could not reach the backend, so nothing was sent.');
    } on TimeoutException {
      throw const ToolFailure(
        'The backend did not answer in time. The email may have been sent: check their Sent '
        'folder before sending it again.',
      );
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final said = response.body.trim();
      throw ToolFailure(said.isEmpty ? 'The backend could not reach Gmail.' : said);
    }
  }
}
