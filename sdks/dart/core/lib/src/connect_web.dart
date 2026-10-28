import 'package:web_socket/web_socket.dart';

/// Whether a socket's handshake carries headers, which a browser's cannot.
const handshakeHeaders = false;

/// Opens [uri] without [headers], which a browser WebSocket cannot send. The credentials are
/// in [uri]'s query instead.
Future<WebSocket> connect(Uri uri, Map<String, String> headers) => WebSocket.connect(uri);
