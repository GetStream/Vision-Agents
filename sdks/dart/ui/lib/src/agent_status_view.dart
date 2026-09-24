import 'package:flutter/material.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// What the agent is doing, in a line.
class AgentStatusView extends StatelessWidget {
  const AgentStatusView({super.key, required this.state});

  final ConversationState state;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final busy = switch (state) {
      Responding() || Working() || Listening() => true,
      Idle() || Ended() => false,
    };
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (busy) ...[
          const SizedBox.square(dimension: 10, child: CircularProgressIndicator(strokeWidth: 1.5)),
          const SizedBox(width: 6),
        ],
        Text(
          label(state),
          style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.outline),
        ),
      ],
    );
  }

  /// The words shown for a state, for a host drawing its own status.
  static String label(ConversationState state) => switch (state) {
    Idle() => 'ready',
    Listening() => 'listening',
    Responding() => 'answering',
    Working(:final skills) when skills.isEmpty => 'thinking',
    Working(:final skills) => 'thinking (${skills.join(', ')})',
    Ended() => 'the conversation ended',
  };
}
