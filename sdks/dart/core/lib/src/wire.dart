import 'dart:convert';

import 'package:http/http.dart' as http;

import 'backend.dart';
import 'errors.dart';

/// One request to the router, decoded.
///
/// The generated operations are written against this rather than against `package:http`, so
/// how a request is authenticated and how a failure is reported live in one hand-written
/// place.
abstract interface class Wire {
  /// Sends one request and returns its decoded JSON body, or null for a response with none.
  ///
  /// A query value that is null is left out rather than sent as the word.
  Future<Object?> send(
    String method,
    String path, {
    required String operation,
    Map<String, String?> query,
    Object? body,
  });
}

/// The wire over `package:http`.
final class HttpWire implements Wire {
  HttpWire(this.backend, this.client);

  final Backend backend;
  final http.Client client;

  @override
  Future<Object?> send(
    String method,
    String path, {
    required String operation,
    Map<String, String?> query = const {},
    Object? body,
  }) async {
    final request = http.Request(method, backend.requestUri(path, query));
    request.headers.addAll(await backend.headers());
    if (body != null) {
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode(body);
    }

    final http.Response response;
    try {
      response = await http.Response.fromStream(await client.send(request));
    } on http.ClientException catch (error) {
      throw TransportException(error);
    }

    if (response.statusCode < 200 || response.statusCode > 299) {
      throw RouterException(response.statusCode, _message(response), operation: operation);
    }
    if (response.bodyBytes.isEmpty) {
      return null;
    }
    try {
      return jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException catch (error) {
      throw UnreadableException('$operation answered with something other than JSON: $error');
    }
  }
}

/// What the router said about a refusal: the `error` of its error body, or the body itself
/// when something in front of it answered instead.
String _message(http.Response response) {
  final text = utf8.decode(response.bodyBytes, allowMalformed: true);
  try {
    final decoded = jsonDecode(text);
    if (decoded is Map && decoded['error'] is String) {
      return decoded['error'] as String;
    }
  } on FormatException {
    // Not JSON: a proxy or a load balancer answered, so what it said is the message.
  }
  return text.trim();
}

/// Reads a timestamp the way the router writes one.
///
/// Go writes RFC 3339 with as many fractional digits as a value needs, up to nine, and none
/// when it needs none, and an offset rather than Z when the host is not on UTC. Dart keeps
/// microseconds, so digits past the sixth are dropped rather than refused.
DateTime? parseRouterDate(String value) {
  final trimmed = value.replaceFirstMapped(RegExp(r'(\.\d{6})\d+'), (match) => match.group(1)!);
  return DateTime.tryParse(trimmed);
}
