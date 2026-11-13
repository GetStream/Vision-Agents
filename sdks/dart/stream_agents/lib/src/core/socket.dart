import 'dart:async';
import 'dart:convert';

import 'package:web_socket/web_socket.dart';

import 'agent_event.dart';
import 'command.dart';
import 'errors.dart';

/// The socket carrying one conversation.
///
/// Hand-written because OpenAPI stops at the upgrade. `package:web_socket` rather than
/// `dart:io` because it is the same code in a browser, and the platform answers the router's
/// pings itself, so there is no keepalive here.
///
/// There is no automatic reconnection, on purpose: `respond` and `tool_result` are not
/// idempotent and the protocol has nothing to resume from, so replaying after a reconnect
/// would duplicate turns and tool results. The stream ends and the caller decides.
final class SessionSocket {
  SessionSocket(this.uri);

  final Uri uri;
  WebSocket? _socket;
  bool _closing = false;

  /// Opens the socket and returns its events.
  ///
  /// The stream is returned rather than kept as a property so it cannot be listened to twice:
  /// two readers of one socket take half the frames each. It ends when the router closes the
  /// socket normally, and ends with a [SocketClosedException] when the connection is lost.
  Future<Stream<AgentEvent>> open() async {
    if (_socket != null) {
      throw StateError('the socket is already open');
    }
    final connecting = WebSocket.connect(uri);
    final WebSocket socket;
    try {
      socket = await connecting.timeout(_opening);
    } on TimeoutException {
      // One that opens after all is closed again, since nobody is waiting for it.
      connecting.then(_closeQuietly).ignore();
      throw TransportException(
        'the session socket did not open within ${_opening.inSeconds} seconds',
      );
    } on WebSocketException catch (error) {
      // Its message is the whole URL, token and all.
      throw TransportException(withoutCredentials('$error'));
    }
    _socket = socket;
    // Closed while it was connecting.
    if (_closing) {
      await _closeQuietly(socket);
      return const Stream.empty();
    }
    return _read(socket);
  }

  /// Sends one command. Throws a [SocketClosedException] when the socket is not open.
  void send(Command command) {
    final socket = _socket;
    if (socket == null || _closing) {
      throw const SocketClosedException(null, 'the socket is not open');
    }
    try {
      socket.sendText(jsonEncode(command.toJson()));
    } on WebSocketConnectionClosed {
      throw const SocketClosedException(null, 'the socket is closed');
    }
  }

  /// Closes the socket. Safe to call more than once, and a close asked for here is not a
  /// failure: the event stream finishes.
  Future<void> close() async {
    if (_closing) {
      return;
    }
    _closing = true;
    if (_socket case final socket?) {
      await _closeQuietly(socket);
    }
  }

  /// Ends the conversation on a frame larger than any the router sends, rather than reading
  /// whatever a broken or hostile server sends next.
  Future<Never> _tooLarge(WebSocket socket) async {
    _closing = true;
    await _closeQuietly(socket);
    throw const SocketClosedException(null, 'the router sent a frame over 16 MiB');
  }

  Future<void> _closeQuietly(WebSocket socket) async {
    try {
      await socket.close(1000, 'closed by the client');
    } on WebSocketConnectionClosed {
      // Already gone, which is what closing wanted.
    }
  }

  Stream<AgentEvent> _read(WebSocket socket) async* {
    await for (final event in socket.events) {
      switch (event) {
        case TextDataReceived(:final text):
          // A string is at least as many bytes as it has code units.
          if (text.length > _largestFrame) {
            await _tooLarge(socket);
          }
          // A frame this SDK cannot read is skipped rather than ending the conversation,
          // which is what the router does with a command it cannot read.
          if (AgentEvent.tryParse(text) case final parsed?) {
            yield parsed;
          }
        case BinaryDataReceived(:final data):
          if (data.length > _largestFrame) {
            await _tooLarge(socket);
          }
          if (AgentEvent.tryParse(utf8.decode(data, allowMalformed: true)) case final parsed?) {
            yield parsed;
          }
        case CloseReceived(:final code, :final reason):
          if (_closing || code == 1000) {
            return;
          }
          throw SocketClosedException(code, reason);
      }
    }
  }
}

/// How long the socket has to open.
const _opening = Duration(seconds: 15);

/// The largest frame read, well past the router's own 5 MiB.
const _largestFrame = 16 * 1024 * 1024;
