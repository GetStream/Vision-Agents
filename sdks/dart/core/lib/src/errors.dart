/// What went wrong talking to the router.
///
/// Sealed so a caller can switch over every case. Four kinds, kept apart on purpose: a request
/// that never got an answer, an answer that said no, an answer this SDK cannot read, and a
/// socket that ended. Never classify one by parsing its message.
sealed class AgentsException implements Exception {
  const AgentsException();
}

/// The router answered, and refused.
final class RouterException extends AgentsException {
  const RouterException(this.status, this.message, {this.operation = ''});

  /// The HTTP status.
  final int status;

  /// What the router said, not a status phrase.
  final String message;

  /// The operation id the request was for, empty when it was not one.
  final String operation;

  /// A 403, which is what a device gets for a path only a backend may take, and for
  /// everything when the app turned its kind of user away.
  bool get isServerSideOnly => status == 403;

  @override
  String toString() =>
      'RouterException: the router answered $status'
      '${operation.isEmpty ? '' : ' to $operation'}: $message';
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
