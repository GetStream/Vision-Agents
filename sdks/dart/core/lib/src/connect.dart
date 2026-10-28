// The socket's handshake, which only a native platform can put headers on.
export 'connect_web.dart' if (dart.library.io) 'connect_io.dart';
