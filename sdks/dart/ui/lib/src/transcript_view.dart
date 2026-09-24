import 'package:flutter/material.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// The conversation, scrolling as it grows.
///
/// It brings no scaffold and no colours of its own beyond the theme, so it drops into
/// whatever the host has.
class TranscriptView extends StatelessWidget {
  /// [turns] is the conversation, oldest first. [bubble] draws one line; omit it for
  /// [TurnBubble].
  const TranscriptView({super.key, required this.turns, this.bubble, this.padding});

  final List<Turn> turns;
  final Widget Function(BuildContext context, Turn turn)? bubble;
  final EdgeInsetsGeometry? padding;

  @override
  Widget build(BuildContext context) {
    final draw = bubble ?? (context, turn) => TurnBubble(turn: turn);
    // Reversed, so the newest turn sits at offset zero: a reply growing several times a
    // second stays in view without a scroll call per delta, and somebody who scrolled up to
    // read is left where they are.
    return ListView.builder(
      reverse: true,
      padding: padding ?? const EdgeInsets.all(16),
      itemCount: turns.length,
      itemBuilder: (context, index) {
        final turn = turns[turns.length - 1 - index];
        return Padding(
          key: ValueKey(turn.id),
          padding: const EdgeInsets.symmetric(vertical: 5),
          child: draw(context, turn),
        );
      },
    );
  }
}

/// One line of the conversation.
class TurnBubble extends StatelessWidget {
  const TurnBubble({super.key, required this.turn});

  final Turn turn;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final agent = turn.speaker.isAgent;
    final name = switch (turn.speaker) {
      ParticipantSpeaker(:final participant?) => participant.display,
      _ => '',
    };
    return Column(
      crossAxisAlignment: agent ? CrossAxisAlignment.start : CrossAxisAlignment.end,
      children: [
        if (name.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(bottom: 3),
            child: Text(
              name,
              style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.outline),
            ),
          ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: agent ? theme.colorScheme.surfaceContainerHighest : theme.colorScheme.primary,
            borderRadius: BorderRadius.circular(16),
          ),
          child: SelectableText(
            turn.text,
            style: TextStyle(
              color: agent ? theme.colorScheme.onSurface : theme.colorScheme.onPrimary,
            ),
          ),
        ),
      ],
    );
  }
}
