import 'dart:async';
import 'dart:ui' show ImageFilter;

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_core_flutter/core.dart';
import 'package:stream_agents/stream_agents.dart';

import 'agent_chat.dart';
import 'assistant.dart';
import 'brand_theme.dart';
import 'sign_in.dart';
import 'voice_screen.dart';

/// One conversation: the messages, the composer, and the way into voice mode.
///
/// A new conversation opens its session as its first message is typed, since opening one takes
/// seconds, and deletes it again if it is left unsent, so opening the app or tapping New chat
/// leaves nothing in the list. A conversation from the list is followed again by its id, and
/// one that ended carries on when it is next asked something. It takes no question until what
/// was said in it has arrived.
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.sessionId, required this.onStarted});

  /// The conversation to carry on, or null for a new one.
  final String? sessionId;

  /// Called with the session a new conversation became, once it is first written or talked
  /// to.
  final ValueChanged<String> onStarted;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  static const _starters = [
    "What's on my calendar today?",
    'Remind me to call mum at 6pm',
    'Plan a weekend in Ohrid',
    'What should I cook tonight?',
  ];

  static final _thinkHarder = ChatOption(
    id: AgentChat.thinkHarder,
    text: 'Think harder',
    description: 'More time to reason before answering',
    // Core has no thinking glyph; its bolt reads as "more effort".
    icon: StreamIcons().bolt,
  );

  /// The conversation's session and channel, opened when the conversation first needs them.
  ///
  /// A question that could not be asked goes back to the composer.
  late final _chat = AgentChat(widget.sessionId, composer: _composer);

  final _composer = ChatComposerController(chatOptions: [_thinkHarder], maxAttachments: 10);

  /// What the conversation shows and what the person does in it. Replies are rated on this
  /// phone only, since the router takes no ratings yet, and are not written again or edited:
  /// the router keeps the conversation in Chat and cannot rewind it.
  late final _conversation = AIConversation(
    transport: _chat,
    composer: _composer,
    approver: _approver,
    actions: AIConversationActions.rate,
  );

  /// What the screen is built from, merged once rather than on every build.
  late final _shown = Listenable.merge([_conversation, _chat.failure, _chat.sessionFailure]);

  /// Why the conversation could not be opened, or null.
  String? _failure;

  /// Whether the list has been told about this new conversation.
  var _announced = false;

  /// How tall the composer floating over the bottom of the conversation is, which the
  /// conversation keeps its latest message clear of.
  var _barHeight = 0.0;

  /// The answers to the tools that ask first, and to the sign-ins the agent's connectors wait
  /// for, as the person signed in here.
  ///
  /// A sign-in opens in the browser, which the router hands back to this app once the provider
  /// has answered; Not now ends the tool call without it.
  late final _approver = AIToolApprover(
    identity: const AIClientIdentity(userId: Assistant.userId),
    decide: (call, decision) async {
      final session = _openSession();
      // The session drops an answer to a call it is not waiting on, such as one asked before
      // the app was last closed, which would leave the card saying it was sent.
      if (!session.approvals.value.any((request) => request.id == call.id)) {
        throw StateError('This question is no longer waiting for an answer.');
      }
      session.decide(call.id, allowed: decision == AIToolApprovalDecision.allowed);
    },
    authorizer: AIToolAuthorizer(
      connect: (call, url) => SignIn.open(url),
      decline: (call) async => _openSession().declineLogin(call.id),
    ),
  );

  /// Stream's components: the composer with the way into voice mode beside the message box.
  /// Its button and its field float over the conversation, so they are frosted rather than
  /// filled.
  ///
  /// Made once, since new builders rebuild every component that reads them.
  late final _components = StreamComponentBuilders(
    extensions: streamAIComponentBuilders(
      composerLeading: (context, props) => _AddButton(props: props),
      composerInput: (context, props) => _Frosted(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.all(context.streamRadius.xxxl)),
        child: ChatComposerInput(props: props),
      ),
      composerTrailing: (context, props) => StreamButton.icon(
        tooltip: 'Talk to Juno',
        onPressed: _talk,
        // Core has no waveform glyph.
        icon: const Icon(Icons.graphic_eq),
      ),
    ),
  );

  @override
  void initState() {
    super.initState();
    _composer.addListener(_draftChanged);
    _conversation.addListener(_conversationChanged);
    _chat.usedSessionId.addListener(_usedChanged);
    if (widget.sessionId != null) _open();
  }

  @override
  void dispose() {
    _composer.removeListener(_draftChanged);
    _conversation
      ..removeListener(_conversationChanged)
      ..dispose();
    _composer.dispose();
    _approver.dispose();
    _chat.usedSessionId.removeListener(_usedChanged);
    unawaited(_leave(_chat));
    super.dispose();
  }

  /// The conversation's session, for answering what it asks.
  ///
  /// Throws a [StateError] while the conversation is not open.
  AgentSession _openSession() {
    final session = _chat.session;
    if (session == null) {
      throw StateError('The conversation is not open yet. Try again in a moment.');
    }
    return session;
  }

  /// Closes [chat], which deletes a new conversation's session that was never used.
  static Future<void> _leave(AgentChat chat) async {
    try {
      await chat.close();
    } on Object catch (error) {
      debugPrint('Closing the conversation failed: $error');
    }
    chat.dispose();
  }

  /// Opens a new conversation's session as its first message is written rather than once it
  /// is sent, since opening one takes seconds.
  void _draftChanged() {
    if (widget.sessionId != null || _chat.session != null) return;
    // After a failed open the next one waits for the question to be sent, rather than each
    // keystroke opening, failing and deleting another session.
    if (_failure != null) return;
    if (_composer.text.isEmpty) return;
    _open();
  }

  /// A question asked again after a failed open clears what that said.
  void _conversationChanged() {
    if (_failure != null && _conversation.status.isAnswering) setState(() => _failure = null);
  }

  /// Tells the list about a new conversation once it is used, written or talked to.
  void _usedChanged() {
    final used = _chat.usedSessionId.value;
    if (widget.sessionId != null || _announced || used == null) return;
    _announced = true;
    widget.onStarted(used);
  }

  /// The conversation's session, opened if it is not yet, and marked used when [use]. Says
  /// why when it cannot be opened.
  Future<AgentSession?> _open({bool use = false}) async {
    if (_failure != null) setState(() => _failure = null);
    try {
      final session = await (use ? _chat.use() : _chat.open());
      // A conversation from the list takes questions from here on.
      if (mounted) setState(() {});
      return session;
    } on AIConversationCancelledException {
      // Left while it opened.
      return null;
    } on Object catch (error) {
      if (mounted) setState(() => _failure = '$error');
      return null;
    }
  }

  /// Switches to voice mode on this conversation, starting a session for a new one.
  Future<void> _talk() async {
    final session = await _open(use: true);
    if (session == null || !mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        fullscreenDialog: true,
        builder: (context) => VoiceScreen(session: session),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return StreamComponentFactory(
      builders: _components,
      child: ListenableBuilder(
        listenable: _shown,
        builder: (context, _) {
          final activity = _conversation.activity;
          final isWelcoming =
              _conversation.messages.isEmpty && _conversation.pendingMessage == null;
          final takesQuestions = _chat.takesQuestions;
          return Stack(
            children: [
              Positioned.fill(
                // The person's messages in bubbles, the agent's as their steps and then their
                // answer. What the person asks shows at once, and Stream's copy takes its
                // place, photos and all. Each question goes to the top as it is asked, and its
                // reply writes itself out below it without moving what is on screen.
                child: NotificationListener<ScrollUpdateNotification>(
                  onNotification: (notification) {
                    if (notification.dragDetails != null) {
                      FocusManager.instance.primaryFocus?.unfocus();
                    }
                    return false;
                  },
                  child: BrandThemeScope(
                    child: AIConversationView.conversation(
                      conversation: _conversation,
                      ui: _chat.ui,
                      footer: _conversation.status.isAnswering && activity?.text != null
                          ? AITypingIndicatorView(text: activity!.text!)
                          : null,
                      emptyContent: Padding(
                        padding: EdgeInsets.only(bottom: _barHeight),
                        child: const _Welcome(),
                      ),
                      padding: EdgeInsets.fromLTRB(16, 0, 16, 16 + _barHeight),
                    ),
                  ),
                ),
              ),
              Positioned(
                left: 0,
                right: 0,
                bottom: 0,
                child: _Measured(
                  onHeight: (height) {
                    if (mounted) setState(() => _barHeight = height);
                  },
                  child: _composerBar(isWelcoming: isWelcoming, takesQuestions: takesQuestions),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  /// Why asking failed: what the conversation could not be opened for, what telling the agent
  /// about the UI of a reply or reading a photo did, why it could not answer, or why the
  /// session's socket stopped.
  String? get _failureText {
    final error = _conversation.error;
    return _failure ??
        _chat.failure.value ??
        (error == null ? null : _describe(error)) ??
        // The socket dropped, such as when the phone changed networks: the tools and the
        // questions they ask stop until the conversation is opened again.
        _chat.sessionFailure.value;
  }

  String _describe(Object error) =>
      error is AIAgentError ? error.localizedMessage(AITranslations.of(context)) : '$error';

  /// What Try again does about [_failureText], or null when there is nothing to try: opens
  /// a conversation from the list that did not open, or asks again what the conversation
  /// could not.
  VoidCallback? get _retry {
    if (_failure != null && !_chat.takesQuestions) return () => unawaited(_open());
    return _conversation.error == null ? null : _conversation.retry;
  }

  /// The starters, why asking failed, and the composer, over the bottom of the conversation,
  /// which fades away under them rather than ending at a bar.
  ///
  /// Until a conversation from the list [takesQuestions], it has no starters and its composer
  /// takes nothing: what is asked before what was said in it arrives would lose its reply.
  Widget _composerBar({required bool isWelcoming, required bool takesQuestions}) {
    final colors = context.streamColorScheme;
    final surface = colors.backgroundApp;
    final failure = _failureText;
    final retry = _retry;
    return DecoratedBox(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [surface.withValues(alpha: 0), surface.withValues(alpha: 0.85), surface],
          stops: const [0, 0.4, 1],
        ),
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (isWelcoming && takesQuestions)
              AISuggestionsView(
                suggestions: _starters,
                onSuggestionSelected: (text) => _conversation.send(text),
              ),
            if (failure != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: _Frosted(
                  shape: const StadiumBorder(),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      spacing: 8,
                      children: [
                        Flexible(
                          child: Text(
                            failure,
                            style: context.streamTextTheme.captionDefault.copyWith(
                              color: colors.accentError,
                            ),
                          ),
                        ),
                        if (retry != null)
                          StreamButton(
                            onPressed: retry,
                            type: .ghost,
                            size: .small,
                            child: const Text('Try again'),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            // The composer's button and field float over the conversation, so they are
            // frosted rather than filled.
            IgnorePointer(
              ignoring: !takesQuestions,
              child: Opacity(
                opacity: takesQuestions ? 1 : 0.5,
                child: ComposerTheme(
                  data: ComposerThemeData(
                    fillColor: colors.backgroundSurfaceStrong.withValues(alpha: 0.6),
                    borderColor: Colors.transparent,
                  ),
                  child: ChatComposer.conversation(
                    conversation: _conversation,
                    enableSpeechToText: true,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// The composer's way to add photos, frosted rather than filled.
class _AddButton extends StatelessWidget {
  const _AddButton({required this.props});

  final ChatComposerLeadingProps props;

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    return _Frosted(
      shape: const CircleBorder(),
      child: ListenableBuilder(
        listenable: props.controller,
        builder: (context, _) => StreamButton.icon(
          icon: Icon(context.streamIcons.plus),
          style: .secondary,
          type: .outline,
          tooltip: AITranslations.of(context).addPhotos,
          onPressed: props.controller.isGenerating ? null : props.openAttachmentSheet,
          themeStyle: StreamButtonThemeStyle(
            tapTargetSize: .shrinkWrap,
            backgroundColor: WidgetStatePropertyAll(
              colors.backgroundSurfaceStrong.withValues(alpha: 0.6),
            ),
            borderColor: const WidgetStatePropertyAll(Colors.transparent),
          ),
        ),
      ),
    );
  }
}

/// A frosted glass that lets the conversation show through, blurred, where something floats
/// over it.
class _Frosted extends StatelessWidget {
  const _Frosted({required this.shape, required this.child});

  final ShapeBorder shape;
  final Widget child;

  @override
  Widget build(BuildContext context) => ClipPath(
    clipper: ShapeBorderClipper(shape: shape),
    child: BackdropFilter(filter: ImageFilter.blur(sigmaX: 16, sigmaY: 16), child: child),
  );
}

/// What a new conversation shows before anything is said.
class _Welcome extends StatelessWidget {
  const _Welcome();

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final spacing = context.streamSpacing;
    return Padding(
      padding: EdgeInsets.all(spacing.xl),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        spacing: spacing.xs,
        children: [
          Icon(context.streamIcons.messageBubbles, size: 44, color: colors.accentPrimary),
          Text(
            _greeting,
            style: text.headingMd.copyWith(color: colors.textPrimary),
            textAlign: TextAlign.center,
          ),
          Text(
            'Ask me anything, show me a photo, or tap the waveform to talk.',
            style: text.bodyDefault.copyWith(color: colors.textSecondary),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  String get _greeting => 'Good ${_partOfDay(DateTime.now().hour)}, ${Assistant.userName}';

  static String _partOfDay(int hour) {
    if (hour < 12) return 'morning';
    if (hour < 18) return 'afternoon';
    return 'evening';
  }
}

/// A box that reports its child's height whenever it changes.
class _Measured extends SingleChildRenderObjectWidget {
  const _Measured({required this.onHeight, required super.child});

  final ValueChanged<double> onHeight;

  @override
  RenderObject createRenderObject(BuildContext context) => _RenderMeasured(onHeight);

  @override
  void updateRenderObject(BuildContext context, _RenderMeasured renderObject) =>
      renderObject.onHeight = onHeight;
}

class _RenderMeasured extends RenderProxyBox {
  _RenderMeasured(this.onHeight);

  ValueChanged<double> onHeight;
  double? _height;

  @override
  void performLayout() {
    super.performLayout();
    final height = size.height;
    if (height == _height) return;
    _height = height;
    WidgetsBinding.instance.addPostFrameCallback((_) => onHeight(height));
  }
}
