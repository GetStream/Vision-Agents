import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'agent_chat.dart';
import 'demo.dart';

/// A written conversation.
///
/// Nothing is transcribed and nothing is spoken, so no call is joined. Everything between
/// hearing a question and answering it is the same as on a call: the same instructions, the
/// same knowledge, the same skills and the same tools running on this phone. What is new in
/// writing is that the reply shows its working, with Stream's AI components: the agent's
/// reasoning as it thinks, each tool call as it runs, and a question when a tool asks first.
class ChatView extends StatefulWidget {
  const ChatView({super.key});

  @override
  State<ChatView> createState() => _ChatViewState();
}

class _ChatViewState extends State<ChatView> {
  AgentChat? _chat;
  String? _failure;

  @override
  void initState() {
    super.initState();
    _open();
  }

  Future<void> _open() async {
    try {
      // A model that streams its reasoning, which is what fills the reply's thinking panel.
      // The call keeps the agent's own model, which answers sooner.
      final session = await Demo.agents
          .agent(Demo.agentName)
          .chat(SessionOptions(llm: 'xai/grok-4.7', tools: Demo.tools));
      final chat = await AgentChat.open(session);
      if (!mounted) {
        await chat.close();
        return;
      }
      setState(() => _chat = chat);
    } on Object catch (error) {
      if (mounted) setState(() => _failure = '$error');
    }
  }

  @override
  void dispose() {
    _chat?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final chat = _chat;
    if (chat != null) return _AgentConversation(chat: chat);
    if (_failure case final failure?) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            'Could not start\n\n$failure',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    return Center(child: StreamLoadingSpinner());
  }
}

class _AgentConversation extends StatefulWidget {
  const _AgentConversation({required this.chat});

  final AgentChat chat;

  @override
  State<_AgentConversation> createState() => _AgentConversationState();
}

class _AgentConversationState extends State<_AgentConversation> {
  static const _starters = [
    'Where is order A-1042?',
    'Can I return order A-1043?',
    "I'd like a refund for order A-1042",
  ];

  final _composer = ChatComposerController();

  /// Answers the questions of the tools that ask first, as the person signed in here.
  ///
  /// The router addresses a question to the person whose message the call answers, and
  /// Stream's components show it only to them. The answer goes back over the session's
  /// socket, which is also what runs the tool once it is allowed.
  late final _approver = AIToolApprover(
    identity: const AIClientIdentity(userId: Demo.userId),
    decide: (call, decision) async => widget.chat.session.decide(
      call.id,
      allowed: decision == AIToolApprovalDecision.allowed,
    ),
  );

  /// Stream's components, with what the person can do under each finished reply: copy it
  /// and rate it, and the composer without its attachment button, since a question goes to
  /// the agent as text. Made once, since new builders rebuild every component that reads
  /// them.
  late final _components = StreamComponentBuilders(
    extensions: streamAIComponentBuilders(
      replyActions: _replyActions,
      composerLeading: (context, props) => ChatComposer.hidden,
    ),
  );

  @override
  void initState() {
    super.initState();
    widget.chat.addListener(_changed);
  }

  @override
  void dispose() {
    widget.chat.removeListener(_changed);
    _composer.dispose();
    _approver.dispose();
    super.dispose();
  }

  /// What the person can do under a finished reply: copy it and rate it. Asking for a reply
  /// again needs the router to replace one, which it does not yet.
  Widget _replyActions(BuildContext context, AIReplyActionsProps props) {
    final chat = widget.chat;
    // Read here, from the chat, so the thumbs change as soon as the rating does.
    return ListenableBuilder(
      listenable: chat,
      builder: (context, _) => AIReplyActionsView(
        text: props.text,
        actions: AIReplyActions(
          rating: chat.ratings[props.message.id],
          onRate: (rating) => chat.rate(props.message.id, rating),
        ),
      ),
    );
  }

  void _changed() {
    _composer.isGenerating = widget.chat.isAnswering;
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final chat = widget.chat;
    final workingOn = chat.workingOn;
    // Here rather than above the app, since the reply actions read this chat.
    return StreamComponentFactory(
      builders: _components,
      child: Column(
        children: [
          Expanded(
            // The person's messages in bubbles, the agent's as their steps and then their
            // answer. Each question goes to the top as it is asked, and its reply writes
            // itself out below it without moving what is on screen.
            child: AIConversationView(
              messages: chat.listMessages,
              isThinking: chat.isThinking,
              approver: _approver,
              loadOlderMessages: chat.hasOlderMessages
                  ? chat.loadOlderMessages
                  : null,
              footer: workingOn == null
                  ? null
                  : AITypingIndicatorView(text: workingOn),
              emptyContent: const _Empty(),
            ),
          ),
          if (chat.messages.isEmpty)
            AISuggestionsView(
              suggestions: _starters,
              onSuggestionSelected: chat.ask,
            ),
          if (chat.failure case final failure?)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                failure,
                style: context.streamTextTheme.captionDefault.copyWith(
                  color: context.streamColorScheme.accentError,
                ),
              ),
            ),
          SafeArea(
            top: false,
            child: ChatComposer(
              controller: _composer,
              enableSpeechToText: true,
              onSendPressed: (text, _, _) => chat.ask(text),
              onStopPressed: chat.stop,
            ),
          ),
        ],
      ),
    );
  }
}

class _Empty extends StatelessWidget {
  const _Empty();

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final spacing = context.streamSpacing;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          context.streamIcons.archive,
          size: 40,
          color: colors.textSecondary,
        ),
        SizedBox(height: spacing.sm),
        Text(
          'Ask about an order',
          style: text.headingSm.copyWith(color: colors.textPrimary),
        ),
        SizedBox(height: spacing.xxs),
        Text(
          'Orders A-1042 and A-1043 are on this account.',
          style: text.bodyDefault.copyWith(color: colors.textSecondary),
        ),
      ],
    );
  }
}
