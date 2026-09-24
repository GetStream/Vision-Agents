import 'package:flutter/material.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'agent_status_view.dart';
import 'composer.dart';
import 'live_value_builder.dart';
import 'transcript_view.dart';

/// A whole conversation: the transcript, what the agent is doing, and somewhere to type.
///
/// The parts are public and work on their own, so a host that wants a different arrangement
/// can take them apart rather than fight this. It opens the socket when it is first built.
/// Closing the session is the host's, since the session outlives any one screen showing it.
class ConversationView extends StatefulWidget {
  const ConversationView({super.key, required this.session, this.prompt = 'Message'});

  final AgentSession session;
  final String prompt;

  @override
  State<ConversationView> createState() => _ConversationViewState();
}

class _ConversationViewState extends State<ConversationView> {
  @override
  void initState() {
    super.initState();
    _start();
  }

  @override
  void didUpdateWidget(ConversationView old) {
    super.didUpdateWidget(old);
    if (old.session != widget.session) {
      _start();
    }
  }

  // A socket that cannot open is reported on `connection`, which is what is drawn.
  Future<void> _start() =>
      widget.session.start().catchError((Object _) {}, test: (error) => error is AgentsException);

  @override
  Widget build(BuildContext context) {
    final session = widget.session;
    final theme = Theme.of(context);
    return LiveValueBuilder(
      value: session.connection,
      builder: (context, connection) => LiveValueBuilder(
        value: session.conversation,
        builder: (context, conversation) => Column(
          children: [
            Expanded(child: TranscriptView(turns: conversation.turns)),
            if (session.failure ?? conversation.failure case final failure?)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(
                  '$failure',
                  style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
                ),
              ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Align(
                alignment: Alignment.centerLeft,
                child: AgentStatusView(state: conversation.state),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Composer(
                prompt: widget.prompt,
                enabled: connection is Connected,
                send: (text) async {
                  try {
                    session.send(text);
                  } on AgentsException {
                    // The socket dropped, which `connection` already shows.
                  }
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
