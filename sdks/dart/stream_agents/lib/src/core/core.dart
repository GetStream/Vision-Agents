/// The state layer and client for Stream Agents, in pure Dart.
///
/// Start from [StreamAgents]. The generated wire types are not exported: everything here is
/// hand-written, so the spec can change without breaking a caller.
library;

export 'agent_event.dart';
export 'agent_session.dart' hide CameraHandler, CameraPicture;
export 'backend.dart';
export 'command.dart';
export 'conversation.dart';
export 'errors.dart' hide withoutCredentials;
export 'guests.dart' show GuestStore, MemoryGuestStore, guestStorageKey;
export 'live_value.dart';
export 'models.dart';
export 'router.dart';
export 'sessions.dart' show Responses, Sessions;
export 'socket.dart';
export 'stream_agents.dart';
export 'tools.dart';
