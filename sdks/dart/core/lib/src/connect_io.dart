import 'dart:io' as io;

import 'package:web_socket/io_web_socket.dart';
import 'package:web_socket/web_socket.dart';

/// Whether a socket's handshake carries headers, which it does everywhere but a browser.
const handshakeHeaders = true;

/// Opens [uri] with [headers] on the handshake, as Swift's and Kotlin's sockets do, so the
/// router reads the socket as the same caller as the requests: a device, not a backend.
///
/// `dart:io` directly because `package:web_socket` takes no headers, and only in this file,
/// which a browser build never compiles.
Future<WebSocket> connect(Uri uri, Map<String, String> headers) async {
  try {
    return IOWebSocket.fromWebSocket(await io.WebSocket.connect('$uri', headers: headers));
  } on io.WebSocketException catch (error) {
    throw WebSocketException(error.message);
  }
}
