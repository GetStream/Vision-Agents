/// One thing a client can do to a running conversation over its socket.
///
/// These are the commands `readCommands` in the router accepts, less `respond`: asking goes
/// through `responses.create`. Everything else a caller might want is a request rather than
/// a frame. A separate type from [AgentEvent], because a
/// frame going in and one coming out have nothing to share.
sealed class Command {
  const Command();

  Map<String, Object?> toJson();
}

/// Speak this without going through the model.
final class SayCommand extends Command {
  const SayCommand(this.text);

  final String text;

  @override
  Map<String, Object?> toJson() => {'type': 'say', 'text': text};
}

/// Abandon the reply in flight.
final class InterruptCommand extends Command {
  const InterruptCommand();

  @override
  Map<String, Object?> toJson() => {'type': 'interrupt'};
}

/// Answer a tool call. One of [output] or [error] says how it went.
final class ToolResultCommand extends Command {
  const ToolResultCommand(
    this.toolCallId, {
    this.output,
    this.error,
    this.requestId = '',
    this.turnId = '',
  });

  final String toolCallId;
  final String? output;
  final String? error;

  /// Repeated from the call when it carried one, so the result cannot be adopted by another
  /// request or turn.
  final String requestId;
  final String turnId;

  @override
  Map<String, Object?> toJson() => {
    'type': 'tool_result',
    'tool_call_id': toolCallId,
    // The router reads both off one struct and treats the empty string as absent, so sending
    // the empty string and sending nothing are the same thing.
    'output': output ?? '',
    'error': error ?? '',
    // Only with a request id: a turn id on its own routes the result to the voice path,
    // which refuses it for a conversation kept in chat.
    if (requestId.isNotEmpty) ...{'request_id': requestId, 'turn_id': turnId},
  };
}

/// A person's answer to a tool call that waited for them. A declined call's step shows
/// [summary]. The call is still answered with [ToolResultCommand] either way, except a
/// connector's sign-in declined with [allowed] false, which the router answers itself.
final class ToolApprovalCommand extends Command {
  const ToolApprovalCommand(
    this.toolCallId, {
    required this.allowed,
    this.summary = '',
    this.requestId = '',
    this.turnId = '',
  });

  final String toolCallId;
  final bool allowed;
  final String summary;
  final String requestId;
  final String turnId;

  @override
  Map<String, Object?> toJson() => {
    'type': 'tool_approval',
    'tool_call_id': toolCallId,
    'allowed': allowed,
    if (summary.isNotEmpty) 'summary': summary,
    if (requestId.isNotEmpty) 'request_id': requestId,
    if (turnId.isNotEmpty) 'turn_id': turnId,
  };
}

/// End the session.
final class CloseCommand extends Command {
  const CloseCommand();

  @override
  Map<String, Object?> toJson() => {'type': 'close'};
}
