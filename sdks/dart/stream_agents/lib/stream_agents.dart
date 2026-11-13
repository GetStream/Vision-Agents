/// The client for Stream Agents.
///
/// Start from [StreamAgents]. `session.chat()` answers the Stream Chat channel a conversation
/// is kept in, and [VoiceSession] joins the agent's call over Stream Video so the conversation
/// can be spoken.
library;

export 'src/chat/chat.dart';
export 'src/core/core.dart';
export 'src/rtc/rtc.dart';
