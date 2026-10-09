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
      throw _refusalOf(response, operation);
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

/// The most of a body that is not the router's kept as a message: a proxy's error page can
/// run to pages, and a message is for a log line.
const _bodyLimit = 1000;

/// The error a response that was not a success reports.
///
/// Every failure the router answers is `{"error": {"message", "type", "code", "doc_url"}}`.
/// A proxy's page, an empty body or an older router's `{"error": "..."}` is not, so it keeps
/// its own text as the message, or the status phrase when it has none, and leaves the type,
/// code and doc url null rather than failing to read. The request id is read either way,
/// since a proxy may still pass it on.
RouterException _refusalOf(http.Response response, String operation) {
  final requestId = _text(response.headers['x-request-id']);
  final text = utf8.decode(response.bodyBytes, allowMalformed: true);
  if (_envelopeOf(text) case final error? when _text(error['message']) != null) {
    return RouterException(
      response.statusCode,
      error['message'] as String,
      operation: operation,
      type: switch (_text(error['type'])) {
        null => null,
        final String type => _errorTypeOf(type),
      },
      code: _text(error['code']),
      docUrl: switch (_text(error['doc_url'])) {
        null => null,
        final String url => Uri.tryParse(url),
      },
      requestId: requestId,
    );
  }
  final trimmed = text.trim();
  final message = trimmed.runes.length > _bodyLimit
      ? '${String.fromCharCodes(trimmed.runes.take(_bodyLimit))}…'
      : trimmed;
  return RouterException(
    response.statusCode,
    message.isEmpty ? response.reasonPhrase ?? '' : message,
    operation: operation,
    requestId: requestId,
  );
}

/// The envelope's `error` object, or null when the body is anything else.
Map<String, Object?>? _envelopeOf(String text) {
  try {
    final decoded = jsonDecode(text);
    if (decoded is Map && decoded['error'] is Map) {
      return (decoded['error'] as Map).cast<String, Object?>();
    }
  } on FormatException {
    // Not JSON: a proxy or a load balancer answered.
  }
  return null;
}

RouterErrorType _errorTypeOf(String wire) => switch (wire) {
  'invalid_request' => RouterErrorType.invalidRequest,
  'authentication' => RouterErrorType.authentication,
  'permission' => RouterErrorType.permission,
  'not_found' => RouterErrorType.notFound,
  'method_not_allowed' => RouterErrorType.methodNotAllowed,
  'not_acceptable' => RouterErrorType.notAcceptable,
  'conflict' => RouterErrorType.conflict,
  'gone' => RouterErrorType.gone,
  'payload_too_large' => RouterErrorType.payloadTooLarge,
  'unsupported_media_type' => RouterErrorType.unsupportedMediaType,
  'rate_limited' => RouterErrorType.rateLimited,
  'internal' => RouterErrorType.internal,
  'unavailable' => RouterErrorType.unavailable,
  _ => RouterErrorType.unknown,
};

/// A string with something in it, or null.
String? _text(Object? value) => value is String && value.isNotEmpty ? value : null;

/// Reads a timestamp the way the router writes one.
///
/// Go writes RFC 3339 with as many fractional digits as a value needs, up to nine, and none
/// when it needs none, and an offset rather than Z when the host is not on UTC. Dart keeps
/// microseconds, so digits past the sixth are dropped rather than refused.
DateTime? parseRouterDate(String value) {
  final trimmed = value.replaceFirstMapped(RegExp(r'(\.\d{6})\d+'), (match) => match.group(1)!);
  return DateTime.tryParse(trimmed);
}
