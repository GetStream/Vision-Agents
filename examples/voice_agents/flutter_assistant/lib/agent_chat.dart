import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_chat_flutter_ai_a2ui/stream_chat_flutter_ai_a2ui.dart';
import 'package:stream_agents/stream_agents.dart';

import 'assistant.dart';
import 'photos.dart';
import 'trip_components.dart';

/// Where an [AgentChat] gets its session and its channel, and drops a session it did not use.
///
/// The app's own are the router's, through [Assistant].
class AgentSessions {
  /// Creates the app's own.
  const AgentSessions();

  /// Carries on the conversation [sessionId], or starts a session for a new one.
  Future<AgentSession> start(String? sessionId) => sessionId != null
      ? Assistant.agents.attach(sessionId, tools: Assistant.tools)
      : Assistant.agents
            .agent(Assistant.agentName)
            .chat(
              SessionOptions(
                llm: Assistant.chatModel,
                tools: Assistant.tools,
                returnUrl: Assistant.returnUrl,
              ),
            );

  /// The channel [session] is kept in, watched with its latest [pageSize] messages.
  Future<stream.Channel> channel(AgentSession session, {required int pageSize}) async {
    final channel = await session.chat();
    await channel.watch(messagesPagination: stream.PaginationParams(limit: pageSize));
    return channel;
  }

  /// Deletes the conversation [sessionId].
  Future<void> delete(String sessionId) => Assistant.sessions.delete(sessionId);
}

/// How Stream's [AIConversation] reaches Juno: the session that answers the conversation, and
/// the channel the router keeps it in.
///
/// The router writes everything here. Asking goes through the session, which records the
/// question in the channel and answers it there: the reply's text, and its steps (each round
/// of reasoning and each tool call) as attachments. The session's socket is what runs this
/// phone's tools and carries the person's answers to the ones that ask first, on a call too.
///
/// A new conversation has no session until it is first written or talked to, which opens one.
/// The router keeps a conversation in Chat and cannot rewind it, so a reply is not written
/// again and a message is not edited: asking for either asks the question again.
class AgentChat with AIConversationTransport<AgentMessage> {
  /// Creates the chat of the conversation [sessionId], or of a new one when it is null.
  ///
  /// A question that could not be asked goes back to [composer], the conversation's own.
  AgentChat(this.sessionId, {this._sessions = const AgentSessions(), this._composer});

  /// How many messages are loaded at a time.
  static const _pageSize = 30;

  /// The conversation to carry on, or null for a new one.
  final String? sessionId;

  final AgentSessions _sessions;
  final ChatComposerController? _composer;

  /// The session, once the conversation has one.
  AgentSession? get session => _session;
  AgentSession? _session;

  /// The session once something was asked in it or said on its call, which a new conversation
  /// joins the list with. Null until then.
  ValueListenable<String?> get usedSessionId => _usedSessionId;
  final _usedSessionId = ValueNotifier<String?>(null);

  /// Why the session's socket stopped, or null: it dropped, such as when the phone changed
  /// networks, after which the tools and the questions they ask stop until the conversation
  /// is opened again.
  ///
  /// A reply the agent could not write is the conversation's error instead, as [failedReply]
  /// says it.
  ValueListenable<String?> get sessionFailure => _sessionFailure;
  final _sessionFailure = ValueNotifier<String?>(null);

  /// Why something could not be asked besides what the conversation says of its own
  /// requests, or null: telling the agent what the person did in a reply's UI failed, or a
  /// photo did not read. Null again once something is next asked.
  ValueListenable<String?> get failure => _failure;
  final _failure = ValueNotifier<String?>(null);

  /// Each reply's thinking in full, from its live updates.
  ///
  /// A stored step only keeps its opening.
  final reasoning = LiveReasoning();

  /// The composer option that asks for more reasoning.
  static const thinkHarder = 'think_harder';

  /// What a reply that failed says, in place of the provider's raw error.
  static const unanswered = "Juno couldn't answer.";

  /// Shows the UI the agent writes for some replies, such as an interactive trip plan, with the
  /// phone's own components, and sends what the person does in it back.
  late final ui = AIUIHost(
    components: TripComponents.all,
    renderers: [A2UIRenderer(onError: (error, _) => debugPrint('A2UI: $error'))],
    send: _answer,
  );

  AIConversationFeed<AgentMessage>? _feed;
  stream.Channel? _channel;
  final _subscriptions = <StreamSubscription<Object?>>[];

  /// The session being opened, so asking while it opens waits for it rather than opening a
  /// second one.
  Future<AgentSession>? _opening;

  /// Whether the channel holds messages older than those loaded.
  var _hasOlderMessages = false;

  /// What the conversation was last told the agent is doing.
  AIAgentActivity? _activity;

  /// Whether the latest reply failed, until the next one starts. The session goes on saying
  /// the agent responds, since nothing ends a reply that failed, which would hide the failure.
  var _unanswered = false;

  /// Callers of [_answer] waiting for the reply on its way to end.
  final _idleWaiters = <Completer<void>>[];

  /// How many times [stop] was called, so a question still on its way to the router can tell
  /// it was stopped meanwhile.
  var _stops = 0;

  /// Whether [close] was called, after which nothing opens, is marked used or is asked, and an
  /// answer still on its way changes nothing.
  var _closed = false;

  /// Whether the conversation can be asked something: a new one at once, and one carried on
  /// once its session is open and the conversation was told what was said in it.
  ///
  /// Asked before then, the conversation would take the last reply of what was said for the
  /// answer, and stop waiting for the real one.
  bool get takesQuestions => !_closed && (sessionId == null || _session != null);

  /// The messages as Stream's AI components show them.
  List<AgentMessage> get listMessages => listed(_channel?.state?.messages ?? const [], reasoning);

  /// The [messages] as Stream's AI components show them, oldest first, without those deleted
  /// and without the replies that ended with nothing to show.
  ///
  /// The router opens a reply, empty, as it takes a question, and ends it as it is when the
  /// reply fails or is stopped. The conversation would take that for the agent's answer, and
  /// forget that it failed.
  static List<AgentMessage> listed(List<stream.Message> messages, LiveReasoning reasoning) => [
    for (final message in messages)
      if (message.deletedAt == null && !message.isUnwrittenReply) AgentMessage(message, reasoning),
  ];

  @override
  void connect(AIConversationFeed<AgentMessage> feed) {
    _feed = feed;
    _publish();
  }

  /// The conversation's session, opened the first time it is asked for: the one it carries on,
  /// or a new one for a new conversation.
  ///
  /// Throws an [AIConversationCancelledException] once the chat is closed.
  Future<AgentSession> open() async {
    _checkNotClosed();
    if (_session case final session?) return session;
    return _opening ??= _start().whenComplete(() => _opening = null);
  }

  /// The session, open and written or talked to.
  ///
  /// Throws an [AIConversationCancelledException] when the chat was closed before it opened:
  /// nobody is told of the session any more, so [close] treats it as one never used.
  Future<AgentSession> use() async {
    final session = await open();
    _checkNotClosed();
    _usedSessionId.value = session.id;
    return session;
  }

  void _checkNotClosed() {
    if (_closed) throw const AIConversationCancelledException();
  }

  /// Carries on the conversation's session, or starts one for a new conversation, and opens
  /// its channel.
  Future<AgentSession> _start() async {
    AgentSession? session;
    try {
      session = await _sessions.start(sessionId);
      final channel = await _sessions.channel(session, pageSize: _pageSize);
      _channel = channel;
      _hasOlderMessages = (channel.state?.messages.length ?? 0) >= _pageSize;
      _follow(session, channel);
      _session = session;
      _publish();
      return session;
    } on Object {
      // A session that opened but whose channel did not is let go of, and a new one deleted,
      // so asking again does not leave another socket open and another conversation listed.
      if (session != null) unawaited(_discard(session));
      rethrow;
    }
  }

  Future<void> _discard(AgentSession session) async {
    try {
      await session.close();
    } on Object catch (error) {
      debugPrint('Closing the session failed: $error');
    }
    if (sessionId != null) return;
    try {
      await _sessions.delete(session.id);
    } on Object {
      // Left in the list, where the person can delete it.
    }
  }

  void _follow(AgentSession session, stream.Channel channel) {
    _subscriptions
      ..add(channel.state!.messagesStream.listen((_) => _publish()))
      // The client's events rather than the channel's list: the list can fold several live
      // updates into one, and each update carries its own window of the reply's thinking.
      ..add(
        channel.client.on(stream.EventType.messageUpdated).listen((event) {
          if (event.cid == channel.cid && event.message != null) {
            reasoning.read(event.message!);
            _publish();
          }
        }),
      )
      ..add(session.conversation.stream.listen((_) => _sessionChanged(session)))
      ..add(session.connection.stream.listen((_) => _sessionChanged(session)))
      // A reply that fails ends with nothing written in the channel: only the socket says so.
      ..add(session.events().listen((event) => _report(event, session)));
    _sessionChanged(session);
  }

  /// Tells the conversation what the channel holds: its messages, each reply with the whole of
  /// its thinking, and whether there are older ones.
  void _publish() {
    final feed = _feed;
    if (feed == null || _channel == null) return;
    feed
      ..updateMessages(listMessages)
      ..updateHasOlderMessages(_hasOlderMessages);
    _checkIdle();
  }

  /// Tells the conversation what the agent is doing, which keeps it waiting for a reply that
  /// takes long, such as one that runs a skill.
  void _sessionChanged(AgentSession session) {
    _sessionFailure.value = session.failure?.toString();
    if (!_unanswered) _tell(activityOf(session.state));
    _checkIdle();
  }

  /// Tells the conversation that the agent could not answer, which stays until its next reply
  /// starts.
  void _report(AgentEvent event, AgentSession session) {
    if (failedReply(event) case final failed?) {
      _unanswered = true;
      // Told even after an earlier failure, which the conversation forgot when it asked again.
      _activity = failed;
      _feed?.updateActivity(failed);
      _checkIdle();
    } else if (event.kind == AgentEventKind.responding) {
      _unanswered = false;
      _sessionChanged(session);
    }
  }

  void _tell(AIAgentActivity? activity) {
    if (activity == _activity) return;
    _activity = activity;
    _feed?.updateActivity(activity);
  }

  /// What the agent is doing in [state]: thinking while it answers, or working on its skills,
  /// which the conversation keeps waiting for. Null when it is doing nothing.
  static AIAgentActivity? activityOf(ConversationState state) => switch (state) {
    Responding() => const AIAgentActivity(state: AIAgentActivityState.thinking),
    Working(:final skills) when skills.isEmpty => const AIAgentActivity(
      state: AIAgentActivityState.thinking,
    ),
    Working(:final skills) => AIAgentActivity(
      state: AIAgentActivityState.checkingExternalSources,
      text: workingOnSkills(skills),
    ),
    _ => null,
  };

  /// What the agent was doing in [event] when it could not answer, or null for any other event.
  ///
  /// The router's reason is a provider's raw error, with its billing links and response body,
  /// so it isn't shown: [unanswered] is.
  static AIAgentActivity? failedReply(AgentEvent event) {
    if (event.kind != AgentEventKind.error || event['context'] != 'llm') return null;
    return const AIAgentActivity(state: AIAgentActivityState.error, text: unanswered);
  }

  /// What the agent is doing while it works on [skills], such as "Working on web search".
  static String workingOnSkills(List<String> skills) =>
      'Working on ${skills.map((skill) => skill.replaceAll('_', ' ')).join(' and ')}';

  /// The reasoning effort to ask with for the composer option the person picked, or null.
  static String? reasoningEffortOf(ChatOption? option) => option?.id == thinkHarder ? 'high' : null;

  /// Asks, with the photos the composer holds and, for Think harder, more time to reason.
  ///
  /// Throws when it could not be asked, which the conversation shows with a way to try again.
  /// A question with a photo that does not read, or one stopped before the router had it, is
  /// not asked: it goes back to the composer, without that photo, and the conversation
  /// forgets it.
  @override
  Future<void> send(AIConversationRequest<AgentMessage> request) async {
    _checkNotClosed();
    _failure.value = null;
    final stops = _stops;
    final session = await open();
    _checkNotStopped(stops, request);
    final List<AgentImage> images;
    try {
      images = await Photos.sources(request.attachments);
    } on PhotoFailure catch (failure) {
      _checkNotClosed();
      _restore(request, without: failure.files);
      _failure.value = '$failure';
      throw const AIConversationCancelledException();
    }
    _checkNotStopped(stops, request);
    // Only now, so a new conversation whose first question was never asked is not listed.
    _usedSessionId.value = session.id;
    await session.responses.create(
      request.text,
      images: images,
      reasoningEffort: reasoningEffortOf(request.chatOption),
    );
    // Stopped while the router took the question: its reply starts only now, so stop it.
    if (stops != _stops) await stop();
  }

  /// Gives up on [request] when the chat was closed, or the person stopped it, since it was
  /// sent: stopped, it goes back to the composer.
  void _checkNotStopped(int stops, AIConversationRequest<AgentMessage> request) {
    _checkNotClosed();
    if (stops == _stops) return;
    _restore(request);
    throw const AIConversationCancelledException();
  }

  /// Puts a question that was not asked back in the composer, which emptied as it was sent,
  /// [without] the photos that did not read, unless something new was written there since.
  void _restore(AIConversationRequest<AgentMessage> request, {List<XFile> without = const []}) {
    final composer = _composer;
    if (composer == null || composer.hasContent) return;
    // Asking again for a reply has nothing of the person's to put back.
    if (request.kind != AIConversationRequestKind.message) return;
    composer.textEditingController.text = request.text;
    composer.addAttachments(request.attachments.where((photo) => !without.contains(photo)));
    if (request.chatOption case final option?) composer.selectChatOption(option);
  }

  /// Tells the agent what the person did in the UI of a reply, such as tapping a place on its
  /// map: their message says it in words, and carries the action for the agent.
  ///
  /// The router takes one response at a time, so an action taken while a reply is still
  /// being written waits for it to end rather than failing.
  Future<void> _answer(AIUIAction action) async {
    await _whenIdle();
    if (_closed) return;
    _failure.value = null;
    try {
      final session = await use();
      await session.responses.create(action.label, uiAction: action.toJson());
    } on Object catch (error) {
      if (_closed) return;
      _failure.value = '$error';
      rethrow;
    }
  }

  /// Whether a reply is on its way, from the first thought to the last word.
  bool get _isAnswering {
    final state = _session?.state;
    if (!_unanswered && (state is Responding || state is Working)) return true;
    final messages = _channel?.state?.messages ?? const <stream.Message>[];
    return messages.lastWhereOrNull((m) => m.isAgentReply)?.isGenerating ?? false;
  }

  /// Completes once no reply is on its way, or this chat is closed.
  Future<void> _whenIdle() {
    if (!_isAnswering || _closed) return Future.value();
    final idle = Completer<void>();
    _idleWaiters.add(idle);
    return idle.future;
  }

  void _checkIdle() {
    if (_isAnswering && !_closed) return;
    for (final waiter in _idleWaiters) {
      waiter.complete();
    }
    _idleWaiters.clear();
  }

  /// Stops the reply in flight, and a question still on its way to the router, which is then
  /// not asked. What was written so far is kept.
  @override
  Future<void> stop() async {
    _stops++;
    try {
      _session?.interrupt();
    } on SocketClosedException {
      // Nothing is answering: the reply in flight stopped with the socket.
    }
  }

  /// Loads the messages before the oldest one loaded.
  @override
  Future<void> loadOlderMessages() async {
    final channel = _channel;
    final first = channel?.state?.messages.firstOrNull;
    if (channel == null || first == null) return;
    final state = await channel.query(
      messagesPagination: stream.PaginationParams(lessThan: first.id, limit: _pageSize),
    );
    if (_closed) return;
    _hasOlderMessages = (state.messages?.length ?? 0) >= _pageSize;
    _publish();
  }

  /// Stops following the channel and closes the session, which ends it on the router. A new
  /// conversation's that was opened while its first message was written, and never used, is
  /// deleted, so it is not left in the list.
  ///
  /// Call [dispose] once it completes.
  Future<void> close() async {
    _closed = true;
    _checkIdle();
    if (_opening case final opening?) {
      try {
        await opening;
      } on Object {
        // Let go of already.
      }
    }
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    // Let go of before the session closes, which can take seconds: the client keeps a watched
    // channel until it is disposed, and hands that one to the next chat on this conversation.
    _channel?.dispose();
    _channel = null;
    final session = _session;
    if (session == null) return;
    try {
      await session.close();
    } on Object catch (error) {
      debugPrint('Closing the conversation failed: $error');
    }
    if (sessionId != null || _usedSessionId.value != null) return;
    // Tried whether or not the close went through.
    try {
      await _sessions.delete(session.id);
    } on Object {
      // Left in the list, where the person can delete it.
    }
  }

  /// Releases what the chat holds, once [close] has completed.
  void dispose() {
    ui.dispose();
    _usedSessionId.dispose();
    _sessionFailure.dispose();
    _failure.dispose();
  }
}

extension on List<stream.Message> {
  stream.Message? lastWhereOrNull(bool Function(stream.Message) test) {
    for (var i = length - 1; i >= 0; i--) {
      if (test(this[i])) return this[i];
    }
    return null;
  }
}

extension AgentReply on stream.Message {
  /// Whether the agent wrote this message, which Stream's AI components render as a reply.
  bool get isAgentReply => extraData['ai_generated'] == true;

  /// Whether the agent is still writing it.
  bool get isGenerating => extraData['generating'] == true;

  /// Whether this is a reply the agent ended without writing anything: no text, no steps, no
  /// files and no UI.
  bool get isUnwrittenReply =>
      isAgentReply &&
      !isGenerating &&
      (text ?? '').isEmpty &&
      attachments.isEmpty &&
      AIUIDocument.fromCustomData(extraData) == null;

  /// The reply's steps, in the order they happened.
  List<AIMessagePart> get steps => AIMessagePart.parseAll([
    for (final attachment in attachments)
      if (attachment.rawType case final type?)
        (type: type, payload: {'id': attachment.id, ...attachment.extraData}),
  ]);
}

/// A message of the conversation as Stream's AI components show it.
class AgentMessage with AIMessage {
  AgentMessage(this.message, this._reasoning) : parts = message.steps;

  final stream.Message message;
  final LiveReasoning _reasoning;

  @override
  final List<AIMessagePart> parts;

  @override
  String get id => message.id;

  @override
  AIMessageRole get role => message.isAgentReply ? AIMessageRole.assistant : AIMessageRole.user;

  @override
  String get text => message.text ?? '';

  @override
  bool get isGenerating => message.isGenerating;

  /// The UI the agent wrote for the reply, which shows in place of its text.
  @override
  AIUIDocument? get uiDocument => AIUIDocument.fromCustomData(message.extraData);

  @override
  List<AIMessageAttachment> get attachments => [
    for (final attachment in message.attachments)
      if (attachment.type == 'image' && attachment.imageUrl != null)
        AIMessageAttachment.image(Uri.parse(attachment.imageUrl!), name: attachment.title)
      else if (attachment.type == 'file' && attachment.assetUrl != null)
        AIMessageAttachment.file(
          Uri.parse(attachment.assetUrl!),
          name: attachment.title,
          size: attachment.fileSize,
          mimeType: attachment.mimeType,
        ),
  ];

  /// The whole of [step]'s thinking, from the live updates while the reply was written.
  ///
  /// A stored step keeps only its opening.
  @override
  String? reasoningText(AIReasoningPart step) => _reasoning[(message: message.id, step: step.id)];
}
