import 'dart:async';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_core_flutter/core.dart';
import 'package:stream_agents/stream_agents.dart';

import 'agent_chat.dart';
import 'demo.dart';
import 'notice.dart';

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

  // Whether a session is being opened, so a second tap on Try again opens no second one.
  var _opening = false;

  @override
  void initState() {
    super.initState();
    _open();
  }

  Future<void> _open() async {
    if (_opening) return;
    _opening = true;
    if (_failure != null) setState(() => _failure = null);
    try {
      // A model that streams its reasoning, which is what fills the reply's thinking panel.
      // The call keeps the agent's own model, which answers sooner.
      final session = await Demo.agents
          .agent(Demo.agentName)
          .chat(SessionOptions(llm: 'xai/grok-4.7', tools: Demo.tools));
      final chat = await AgentChat.open(session);
      if (!mounted) {
        await chat.close();
        chat.dispose();
        return;
      }
      setState(() => _chat = chat);
    } on Object catch (error) {
      if (mounted) setState(() => _failure = '$error');
    } finally {
      _opening = false;
    }
  }

  @override
  void dispose() {
    final chat = _chat;
    if (chat != null) unawaited(_leave(chat));
    super.dispose();
  }

  static Future<void> _leave(AgentChat chat) async {
    try {
      await chat.close();
    } on Object catch (error) {
      debugPrint('Closing the conversation failed: $error');
    }
    chat.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final chat = _chat;
    if (chat != null) return _AgentConversation(chat: chat);
    if (_failure case final failure?) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Notice(
              icon: context.streamIcons.exclamationTriangleFill,
              title: 'Could not start',
              message: failure,
            ),
            StreamButton(onPressed: _open, child: const Text('Try again')),
          ],
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

  late final _conversation = AIConversation(
    transport: widget.chat,
    composer: _composer,
    approver: _approver,
    actions: AIConversationActions.rate,
  );

  late final _shown = Listenable.merge([
    _conversation,
    widget.chat.sessionFailure,
  ]);

  // Answers the questions of the tools that ask first, as the person signed in here.
  //
  // The router addresses a question to the person whose message the call answers, and
  // Stream's components show it only to them. The answer goes back over the session's
  // socket, which is also what runs the tool once it is allowed.
  late final _approver = AIToolApprover(
    identity: const AIClientIdentity(userId: Demo.userId),
    decide: (call, decision) async {
      final session = widget.chat.session;
      // The session drops an answer to a question it is not waiting on, such as one asked
      // before its socket dropped, which would leave the step saying it was sent while the
      // tool never runs.
      if (!session.approvals.value.any((request) => request.id == call.id)) {
        throw StateError('This question is no longer waiting for an answer.');
      }
      session.decide(
        call.id,
        allowed: decision == AIToolApprovalDecision.allowed,
      );
    },
  );

  // Stream's components, with the composer without its attachment button: a question goes to
  // the agent as text, and dictating one stays. Copy, share and rate show under each finished
  // reply through AIConversationActions.rate.
  late final _components = StreamComponentBuilders(
    extensions: streamAIComponentBuilders(
      composerLeading: (context, props) => ChatComposer.hidden,
    ),
  );

  @override
  void dispose() {
    _conversation.dispose();
    _composer.dispose();
    _approver.dispose();
    super.dispose();
  }

  String _describe(BuildContext context, Object error) => error is AIAgentError
      ? error.localizedMessage(AITranslations.of(context))
      : '$error';

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    return StreamComponentFactory(
      builders: _components,
      child: ListenableBuilder(
        listenable: _shown,
        builder: (context, _) {
          final activity = _conversation.activity;
          final error = _conversation.error;
          final failure = error == null
              ? widget.chat.sessionFailure.value
              : _describe(context, error);
          final isWelcoming =
              _conversation.messages.isEmpty &&
              _conversation.pendingMessage == null;
          return Column(
            children: [
              Expanded(
                child: AIConversationView.conversation(
                  conversation: _conversation,
                  footer:
                      _conversation.status.isAnswering && activity?.text != null
                      ? AITypingIndicatorView(text: activity!.text!)
                      : null,
                  emptyContent: Notice(
                    icon: context.streamIcons.archive,
                    title: 'Ask about an order',
                    message: 'Orders A-1042 and A-1043 are on this account.',
                  ),
                ),
              ),
              if (isWelcoming)
                AISuggestionsView(
                  suggestions: _starters,
                  onSuggestionSelected: _conversation.send,
                ),
              if (failure != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    spacing: 8,
                    children: [
                      Flexible(
                        child: Text(
                          failure,
                          style: context.streamTextTheme.captionDefault
                              .copyWith(color: colors.accentError),
                        ),
                      ),
                      if (error != null)
                        StreamButton(
                          onPressed: _conversation.retry,
                          type: .ghost,
                          size: .small,
                          child: const Text('Try again'),
                        ),
                    ],
                  ),
                ),
              SafeArea(
                top: false,
                child: ChatComposer.conversation(
                  conversation: _conversation,
                  enableSpeechToText: true,
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
