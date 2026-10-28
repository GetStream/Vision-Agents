import 'package:web_socket/web_socket.dart';

/// Whether a socket can send headers on its handshake, which a browser's cannot.
const canSetSocketHeaders = false;

/// Opens [uri] without [headers], which a browser WebSocket cannot send. The credentials are
/// in [uri]'s query instead.
Future<WebSocket> connect(Uri uri, Map<String, String> headers) => WebSocket.connect(uri);
