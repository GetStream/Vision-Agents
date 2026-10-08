/// What went wrong talking to the router.
///
/// Sealed so a caller can switch over every case. Four kinds, kept apart on purpose: a request
/// that never got an answer, an answer that said no, an answer this SDK cannot read, and a
/// socket that ended. Never classify one by parsing its message.
sealed class AgentsException implements Exception {
  const AgentsException();
}

/// The kind of failure the router named, which decides the status it answered with.
enum RouterErrorType {
  /// 400.
  invalidRequest,

  /// 401.
  authentication,

  /// 403.
  permission,

  /// 404.
  notFound,

  /// 405.
  methodNotAllowed,

  /// 406.
  notAcceptable,

  /// 409.
  conflict,

  /// 410.
  gone,

  /// 413.
  payloadTooLarge,

  /// 415.
  unsupportedMediaType,

  /// 429.
  rateLimited,

  /// 500, which says only "something went wrong": quote [RouterException.requestId].
  internal,

  /// 503.
  unavailable,

  /// A type this SDK has never heard of.
  unknown,
}

/// The router answered, and refused.
final class RouterException extends AgentsException {
  const RouterException(
    this.status,
    this.message, {
    this.operation = '',
    this.type,
    this.code,
    this.docUrl,
    this.requestId,
  });

  /// The HTTP status.
  final int status;

  /// What went wrong, for a person to read: the router's own words, the text of a body
  /// something in front of it answered with, or the status phrase when there was neither.
  /// Its wording may change; branch on [code].
  final String message;

  /// The operation id the request was for, empty when it was not one.
  final String operation;

  /// The kind of failure. Null when the body was not the router's.
  final RouterErrorType? type;

  /// What went wrong, for a program to branch on: `not_configured`, `validation_failed`,
  /// `session_not_found` and more as the router adds them, so expect one this SDK does not
  /// know. Null when the body was not the router's.
  final String? code;

  /// Where [code] is explained.
  final Uri? docUrl;

  /// The response's `X-Request-Id`, which is what to quote to support: a 500 says only
  /// "something went wrong", and this is how the rest of it is found.
  final String? requestId;

  /// A 403, which is what a device gets for a path only a backend may take, and for
  /// everything when the app turned its kind of user away.
  bool get isServerSideOnly => status == 403;

  @override
  String toString() =>
      'RouterException: the router answered $status'
      '${operation.isEmpty ? '' : ' to $operation'}: $message'
      '${code == null ? '' : ' ($code)'}'
      '${requestId == null ? '' : ' [request $requestId]'}';
}

/// The request never got an answer.
final class TransportException extends AgentsException {
  const TransportException(this.cause);

  final Object cause;

  @override
  String toString() => 'TransportException: could not reach the router: $cause';
}

/// The router answered with something this SDK cannot read.
final class UnreadableException extends AgentsException {
  const UnreadableException(this.what);

  final String what;

  @override
  String toString() => "UnreadableException: could not read the router's answer: $what";
}

/// The session socket ended before the session did.
///
/// A socket the router closed normally, because the session ended, is not one of these: the
/// event stream simply finishes.
final class SocketClosedException extends AgentsException {
  const SocketClosedException(this.code, this.reason);

  /// The WebSocket close code, or null when the connection dropped without one.
  final int? code;
  final String reason;

  @override
  String toString() =>
      'SocketClosedException: the session socket closed (${code ?? 'no code'})'
      '${reason.isEmpty ? '' : ': $reason'}';
}
