import 'models.dart';

/// One thing a client can do to a running conversation over its socket.
///
/// These are the commands `readCommands` in the router accepts. Everything else a caller
/// might want is a request rather than a frame. A separate type from [AgentEvent], because a
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

/// Answer this as though it had been heard.
final class RespondCommand extends Command {
  const RespondCommand(this.text, {this.images = const []});

  final String text;

  /// Handed to the vision skill, which reports what it finds back into the conversation.
  final List<AgentImage> images;

  @override
  Map<String, Object?> toJson() => {
    'type': 'respond',
    'text': text,
    if (images.isNotEmpty)
      'images': [
        for (final image in images) {'url': image.url, 'detail': ?image.detail},
      ],
  };
}

/// Abandon the reply in flight.
final class InterruptCommand extends Command {
  const InterruptCommand();

  @override
  Map<String, Object?> toJson() => {'type': 'interrupt'};
}

/// Replace the system prompt, from the next turn on.
final class InstructionsCommand extends Command {
  const InstructionsCommand(this.instructions);

  final String instructions;

  @override
  Map<String, Object?> toJson() => {'type': 'instructions', 'instructions': instructions};
}

/// Answer a tool call. One of [output] or [error] says how it went.
final class ToolResultCommand extends Command {
  const ToolResultCommand(
    this.toolCallId, {
    this.output,
    this.error,
    this.commandId = '',
    this.turnId = '',
  });

  final String toolCallId;
  final String? output;
  final String? error;

  /// Repeated from the call when it carried one, so the result cannot be adopted by another
  /// command or turn.
  final String commandId;
  final String turnId;

  @override
  Map<String, Object?> toJson() => {
    'type': 'tool_result',
    'tool_call_id': toolCallId,
    // The router reads both off one struct and treats the empty string as absent, so sending
    // the empty string and sending nothing are the same thing.
    'output': output ?? '',
    'error': error ?? '',
    // Only with a command id: a turn id on its own routes the result to the voice path,
    // which refuses it for a conversation kept in chat.
    if (commandId.isNotEmpty) ...{'command_id': commandId, 'turn_id': turnId},
  };
}

/// End the session.
final class CloseCommand extends Command {
  const CloseCommand();

  @override
  Map<String, Object?> toJson() => {'type': 'close'};
}
