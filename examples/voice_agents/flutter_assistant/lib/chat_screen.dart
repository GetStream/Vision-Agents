import 'dart:async';
import 'dart:ui' show ImageFilter;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'agent_chat.dart';
import 'assistant.dart';
import 'voice_screen.dart';

/// One conversation: the messages, the composer, and the way into voice mode.
///
/// A new conversation has no session until its first message, so opening the app or tapping
/// New chat leaves nothing in the list. A conversation from the list is followed again by its
/// id, and one that ended carries on when it is next asked something.
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.sessionId, required this.onStarted});

  /// The conversation to carry on, or null for a new one.
  final String? sessionId;

  /// Says which session a new conversation became, once it is first written or talked to.
  final ValueChanged<String> onStarted;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  static const _starters = [
    "What's on my calendar today?",
    'Remind me to call mum at 6pm',
    'Plan a weekend in Lisbon',
    'What should I cook tonight?',
  ];

  static const _thinkHarder = ChatOption(
    id: 'think_harder',
    text: 'Think harder',
    description: 'More time to reason before answering',
    icon: Icons.psychology_outlined,
  );

  AgentChat? _chat;
  String? _failure;

  /// What the person just asked, shown at once with its photos until Stream's copy of their
  /// message arrives and takes its place: opening a conversation, uploading its photos and
  /// writing the message take seconds together.
  _Asked? _asking;

  /// The session being opened, so asking while it opens waits for it rather than opening a
  /// second one.
  Future<AgentChat?>? _opening;

  /// Whether the list has been told about this new conversation.
  var _announced = false;

  /// A new conversation's session, opened while the person wrote their first message and not
  /// used yet. Leaving without using it deletes it, so it is not left in the list.
  String? _unused;

  /// How tall the composer floating over the bottom of the conversation is, which the
  /// conversation keeps its latest message clear of.
  var _barHeight = 0.0;

  final _composer = ChatComposerController(
    chatOptions: const [_thinkHarder],
    maxAttachments: 10,
  );

  /// Answers the questions of the tools that ask first, as the person signed in here.
  late final _approver = AIToolApprover(
    identity: const AIClientIdentity(userId: Assistant.userId),
    decide: (call, decision) async {
      final chat = _chat;
      if (chat == null) throw StateError('the conversation is not open');
      chat.session.decide(
        call.id,
        allowed: decision == AIToolApprovalDecision.allowed,
      );
    },
  );

  /// Stream's components, with what the person can do under each finished reply. Made once,
  /// since new builders rebuild every component that reads them.
  late final _components = AIComponentBuilders(replyActions: _replyActions);

  @override
  void initState() {
    super.initState();
    _composer.addListener(_draftChanged);
    if (widget.sessionId != null) _open();
  }

  @override
  void dispose() {
    _composer
      ..removeListener(_draftChanged)
      ..dispose();
    _approver.dispose();
    final chat = _chat;
    chat?.removeListener(_chatChanged);
    unawaited(_leave(chat, _unused));
    super.dispose();
  }

  /// Closes the conversation, and deletes a session that was opened but never used.
  static Future<void> _leave(AgentChat? chat, String? unused) async {
    await chat?.close();
    chat?.dispose();
    if (unused == null) return;
    try {
      await Assistant.sessions.delete(unused);
    } on Object {
      // Left in the list, where the person can delete it.
    }
  }

  /// Opening a session takes seconds, so a new conversation's opens as its first message is
  /// written rather than once it is sent.
  void _draftChanged() {
    if (widget.sessionId != null ||
        _chat != null ||
        _opening != null ||
        _composer.text.isEmpty) {
      return;
    }
    _open().then((chat) {
      if (chat != null && !_announced) _unused = chat.session.id;
    });
  }

  void _chatChanged() {
    final chat = _chat!;
    _composer.isGenerating = chat.isAnswering;
    final asked = _asking;
    final after = asked?.after;
    setState(() {
      if (after != null && chat.personMessageCount > after) _asking = null;
    });
  }

  /// The conversation's chat, starting a session for a new one.
  Future<AgentChat?> _open() async {
    if (_chat case final chat?) return chat;
    if (_opening case final opening?) return opening;
    final opening = _opening = _start();
    final opened = await opening;
    _opening = null;
    return opened;
  }

  /// Carries on the conversation's session, or starts one for a new conversation, and opens
  /// its channel.
  Future<AgentChat?> _start() async {
    _failure = null;
    final sessionId = widget.sessionId;
    AgentSession? session;
    try {
      session = sessionId != null
          ? await Assistant.agents.attach(sessionId, tools: Assistant.tools)
          : await Assistant.agents
                .agent(Assistant.agentName)
                .chat(
                  SessionOptions(
                    llm: Assistant.chatModel,
                    tools: Assistant.tools,
                  ),
                );
      final chat = await AgentChat.open(session);
      if (!mounted) {
        await _leave(chat, sessionId == null ? session.id : null);
        return null;
      }
      chat.addListener(_chatChanged);
      setState(() => _chat = chat);
      return chat;
    } on Object catch (error) {
      // A session that opened but whose channel did not is let go of, and a new one deleted,
      // so asking again does not leave another socket open and another conversation listed.
      if (session != null) unawaited(_discard(session, isNew: sessionId == null));
      if (mounted) setState(() => _failure = '$error');
      return null;
    }
  }

  static Future<void> _discard(AgentSession session, {required bool isNew}) async {
    try {
      await session.close();
      if (isNew) await Assistant.sessions.delete(session.id);
    } on Object {
      // Left in the list, where the person can delete it.
    }
  }

  Future<void> _send(
    String text, {
    List<XFile> photos = const [],
    bool thinkHarder = false,
  }) async {
    final asked = _Asked(text, photos);
    setState(() => _asking = asked);
    final chat = await _open();
    if (chat == null) {
      if (mounted && identical(_asking, asked)) setState(() => _asking = null);
      return;
    }
    asked.after = chat.personMessageCount;
    _markUsed(chat);
    await chat.ask(text, photos: photos, thinkHarder: thinkHarder);
    if (mounted && chat.failure != null && identical(_asking, asked)) {
      setState(() => _asking = null);
    }
  }

  /// Switches to voice mode on this conversation, starting a session for a new one.
  Future<void> _talk() async {
    final chat = await _open();
    if (chat == null || !mounted) return;
    _markUsed(chat);
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        fullscreenDialog: true,
        builder: (context) => VoiceScreen(chat: chat),
      ),
    );
  }

  /// Tells the list about a new conversation once it is used, written or talked to.
  void _markUsed(AgentChat chat) {
    _unused = null;
    if (widget.sessionId != null || _announced) return;
    _announced = true;
    widget.onStarted(chat.session.id);
  }

  /// What the person can do under a finished reply: copy it and rate it. Asking for a reply
  /// again needs the router to replace one, which it does not yet.
  Widget _replyActions(BuildContext context, AIReplyActionsProps props) {
    final chat = _chat!;
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

  @override
  Widget build(BuildContext context) {
    final chat = _chat;
    final workingOn = chat?.workingOn;
    final isWelcoming = (chat?.messages.isEmpty ?? true) && _asking == null;
    return AIComponentFactory(
      builders: _components,
      child: Stack(
        children: [
          Positioned.fill(
            // The person's messages in bubbles, the agent's as their steps and then their
            // answer. What the person asks shows at once, and Stream's copy takes its place,
            // photos and all. Each question goes to the top as it is asked, and its reply
            // writes itself out below it without moving what is on screen.
            child: NotificationListener<ScrollUpdateNotification>(
              onNotification: (notification) {
                if (notification.dragDetails != null) {
                  FocusManager.instance.primaryFocus?.unfocus();
                }
                return false;
              },
              child: AIConversationView(
                messages: chat?.listMessages ?? const [],
                pendingMessage: _asking?.pending,
                // A question on its way is thought about before the session says anything.
                isThinking:
                    workingOn == null &&
                    ((chat?.isThinking ?? false) || _asking != null),
                approver: chat == null ? null : _approver,
                loadOlderMessages: chat != null && chat.hasOlderMessages
                    ? chat.loadOlderMessages
                    : null,
                footer: workingOn == null
                    ? null
                    : AITypingIndicatorView(text: workingOn),
                emptyContent: Padding(
                  padding: EdgeInsets.only(bottom: _barHeight),
                  child: const _Welcome(),
                ),
                padding: EdgeInsets.fromLTRB(16, 0, 16, 16 + _barHeight),
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
              child: _composerBar(chat, isWelcoming: isWelcoming),
            ),
          ),
        ],
      ),
    );
  }

  /// The starters, why asking failed, and the composer, over the bottom of the conversation,
  /// which fades away under them rather than ending at a bar.
  Widget _composerBar(AgentChat? chat, {required bool isWelcoming}) {
    final theme = Theme.of(context);
    final surface = theme.colorScheme.surface;
    final failure =
        _failure ??
        chat?.failure ??
        chat?.session.conversation.value.failure ??
        // The socket dropped, such as when the phone changed networks: the tools and the
        // questions they ask stop until the conversation is opened again.
        chat?.session.failure?.toString();
    return DecoratedBox(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            surface.withValues(alpha: 0),
            surface.withValues(alpha: 0.85),
            surface,
          ],
          stops: const [0, 0.4, 1],
        ),
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (isWelcoming)
              AISuggestionsView(
                suggestions: _starters,
                onSuggestionSelected: _send,
              ),
            if (failure != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: _Frosted(
                  shape: const StadiumBorder(),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    child: Text(
                      failure,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.error,
                      ),
                    ),
                  ),
                ),
              ),
            // The composer's button and field float over the conversation, so they are
            // frosted rather than filled.
            ComposerTheme(
              data: ComposerThemeData(
                fillColor: theme.colorScheme.surfaceContainerHigh.withValues(
                  alpha: 0.6,
                ),
                borderColor: Colors.transparent,
              ),
              child: ChatComposer(
                controller: _composer,
                factory: _Composer(onTalk: _talk),
                enableSpeechToText: true,
                onSendPressed: (text, option, photos) => _send(
                  text,
                  photos: photos,
                  thinkHarder: option?.id == _thinkHarder.id,
                ),
                onStopPressed: chat?.stop,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// A question on its way, as the person sees it before Stream's copy of it arrives.
class _Asked {
  _Asked(String text, List<XFile> photos)
    : pending = AIPendingMessage(
        text: text,
        attachments: [
          for (final photo in photos)
            AIMessageAttachment.image(Uri.file(photo.path)),
        ],
      );

  /// Its row in the list, which Stream's copy of it takes over.
  final AIPendingMessage pending;

  /// How many of the person's messages the conversation held when it was asked, once it is
  /// open: the next one is this question.
  int? after;
}

/// The composer with the way into voice mode beside the message box. Its button and its field
/// float over the conversation, so they are frosted rather than filled.
class _Composer extends ChatComposerFactory {
  const _Composer({required this.onTalk});

  final VoidCallback onTalk;

  @override
  Widget? buildLeading(BuildContext context, ChatComposerLeadingProps props) {
    final button = super.buildLeading(context, props);
    return button == null
        ? null
        : _Frosted(shape: const CircleBorder(), child: button);
  }

  @override
  Widget buildInput(BuildContext context, ChatComposerInputProps props) =>
      _Frosted(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        child: super.buildInput(context, props),
      );

  @override
  Widget? buildTrailing(BuildContext context, ChatComposerTrailingProps props) =>
      IconButton.filled(
        tooltip: 'Talk to Juno',
        onPressed: onTalk,
        icon: const Icon(Icons.graphic_eq),
      );
}

/// Lets the conversation show through, blurred, where something floats over it.
class _Frosted extends StatelessWidget {
  const _Frosted({required this.shape, required this.child});

  final ShapeBorder shape;
  final Widget child;

  @override
  Widget build(BuildContext context) => ClipPath(
    clipper: ShapeBorderClipper(shape: shape),
    child: BackdropFilter(
      filter: ImageFilter.blur(sigmaX: 16, sigmaY: 16),
      child: child,
    ),
  );
}

/// What a new conversation shows before anything is said.
class _Welcome extends StatelessWidget {
  const _Welcome();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        spacing: 8,
        children: [
          Icon(Icons.auto_awesome, size: 44, color: theme.colorScheme.primary),
          Text(
            _greeting,
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
            textAlign: TextAlign.center,
          ),
          Text(
            'Ask me anything, show me a photo, or tap the waveform to talk.',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  String get _greeting {
    final hour = DateTime.now().hour;
    final part = hour < 12
        ? 'morning'
        : hour < 18
        ? 'afternoon'
        : 'evening';
    return 'Good $part, ${Assistant.userName}';
  }
}

/// Reports its child's height whenever it changes.
class _Measured extends SingleChildRenderObjectWidget {
  const _Measured({required this.onHeight, required super.child});

  final ValueChanged<double> onHeight;

  @override
  RenderObject createRenderObject(BuildContext context) =>
      _RenderMeasured(onHeight);

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
