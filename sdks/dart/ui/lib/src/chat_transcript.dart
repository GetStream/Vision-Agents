import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_core/stream_chat_flutter_core.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'transcript_view.dart';

/// The Stream Chat channel a conversation is kept in, or null for one that is not.
///
/// A session opened with `persistConversation` names its channel as `type:id`. Reading it
/// takes a chat client connected as the person holding the conversation, which the app's
/// backend mints a Chat token for.
Channel? conversationChannel(StreamChatClient client, Session session) {
  final cid = session.conversationId;
  final colon = cid.indexOf(':');
  if (colon <= 0 || colon == cid.length - 1) {
    return null;
  }
  return client.channel(cid.substring(0, colon), id: cid.substring(colon + 1));
}

/// A conversation kept in Stream Chat, as turns.
///
/// The router writes each turn with a `support_message` payload whose `role` says who spoke;
/// a message without one, such as a spoken turn copied in, is the agent's when it was sent
/// by [agentUserId]. A turn still being written has no text yet, and is left out until it
/// has some.
List<Turn> turnsOfMessages(Iterable<Message> messages, {String agentUserId = ''}) => [
  for (final message in messages)
    if (!message.isDeleted && _textOf(message).isNotEmpty)
      Turn(
        id: message.id,
        speaker: _isAgent(message, agentUserId)
            ? Speaker.agent
            : ParticipantSpeaker(
                Participant(userId: message.user?.id ?? '', name: message.user?.name ?? ''),
              ),
        text: _textOf(message),
        at: message.createdAt,
      ),
];

/// A conversation kept in Stream Chat, read from its channel and following new messages.
///
/// Its history is whatever the channel holds, from before this device opened it too, which
/// the session socket alone cannot give. It is only the transcript: sending still goes
/// through the session, so the agent answers.
class ChatTranscriptView extends StatefulWidget {
  const ChatTranscriptView({super.key, required this.channel, this.bubble});

  final Channel channel;
  final Widget Function(BuildContext context, Turn turn)? bubble;

  @override
  State<ChatTranscriptView> createState() => _ChatTranscriptViewState();
}

class _ChatTranscriptViewState extends State<ChatTranscriptView> {
  late Future<void> _watching;

  @override
  void initState() {
    super.initState();
    _watching = widget.channel.watch();
  }

  @override
  void didUpdateWidget(ChatTranscriptView old) {
    super.didUpdateWidget(old);
    if (old.channel != widget.channel) {
      _watching = widget.channel.watch();
    }
  }

  @override
  Widget build(BuildContext context) => FutureBuilder(
    future: _watching,
    builder: (context, watched) {
      final state = widget.channel.state;
      if (watched.hasError) {
        return Center(child: Text('${watched.error}'));
      }
      if (state == null) {
        return const Center(child: CircularProgressIndicator());
      }
      final agent = '${widget.channel.extraData['support_agent_id'] ?? ''}';
      return StreamBuilder(
        stream: state.messagesStream,
        initialData: state.messages,
        builder: (context, messages) => TranscriptView(
          turns: turnsOfMessages(messages.data ?? const [], agentUserId: agent),
          bubble: widget.bubble,
        ),
      );
    },
  );
}

bool _isAgent(Message message, String agentUserId) =>
    switch (message.extraData['support_message']) {
      {'role': final String role} => role == 'assistant',
      _ => agentUserId.isNotEmpty && message.user?.id == agentUserId,
    };

String _textOf(Message message) => switch (message.extraData['support_message']) {
  {'text': final String text} when text.isNotEmpty => text,
  _ => message.text ?? '',
};
