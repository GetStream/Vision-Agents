import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_chat_flutter_ai_a2ui/stream_chat_flutter_ai_a2ui.dart';
import 'package:stream_agents/stream_agents.dart';

import 'assistant.dart';
import 'device_tools.dart';
import 'openui_renderer.dart';
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

  /// Fast-lane 0ms WebSocket UI updates overlaid ahead of Stream Chat's 150ms ephemeral tick.
  final liveUI = LiveUI();

  /// The tables each seat selector in the conversation offers, oldest first, by the reply it
  /// is in: the same card shown again is the same offer, and two venues' tables of the same ids
  /// are two.
  final _offers = <String, List<DiningTableItem>>{};

  /// The bookings made from the seat selectors, by the offer each was made from.
  ValueListenable<Map<String, Reservation>> get reservations => _reservations;
  final _reservations = ValueNotifier<Map<String, Reservation>>(const {});

  /// The tool a card brings while it is being added, which a question waits for, so the reply
  /// it asks for can call it.
  Future<void>? _adding;

  /// The tools the cards in the conversation brought to its session, by name, so a card shown
  /// again, such as after scrolling back to it, does not bring its tool twice.
  final _brought = <String>{};

  /// The composer option that asks for more reasoning.
  static const thinkHarder = 'think_harder';

  /// What a reply that failed says, in place of the provider's raw error.
  static const unanswered = "Juno couldn't answer.";

  /// Shows the UI the agent writes for some replies, such as an interactive trip plan, with the
  /// phone's own components, and sends what the person does in it back.
  late final ui = AIUIHost(
    components: TripComponents.all,
    renderers: [
      OpenUIRenderer(onError: (error, _) => debugPrint('OpenUI: $error')),
      A2UIRenderer(onError: (error, _) => debugPrint('A2UI: $error')),
    ],
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
  List<AgentMessage> get listMessages =>
      listed(liveUI.overlay(_channel?.state?.messages ?? const []), reasoning, shown: _shown);

  /// Each message as it was last shown. A changed message is a new one, so one that did not
  /// change is not parsed again as every streamed word arrives.
  final _shown = Expando<AgentMessage>();

  /// The [messages] as Stream's AI components show them, oldest first, without those deleted
  /// and without the replies that ended with nothing to show, reusing those in [shown].
  ///
  /// The router opens a reply, empty, as it takes a question, and ends it as it is when the
  /// reply fails or is stopped. The conversation would take that for the agent's answer, and
  /// forget that it failed.
  static List<AgentMessage> listed(
    List<stream.Message> messages,
    LiveReasoning reasoning, {
    Expando<AgentMessage>? shown,
  }) {
    final known = shown ?? Expando<AgentMessage>();
    return [
      for (final message in messages)
        if (message.deletedAt == null && !message.isUnwrittenReply)
          known[message] ??= AgentMessage(message, reasoning),
    ];
  }

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
          // The rest of an update reaches the channel's messages, which publish it.
          final message = event.message;
          if (event.cid == channel.cid && message != null && _readReasoning(message)) _publish();
        }),
      )
      ..add(session.conversation.stream.listen((_) => _sessionChanged(session)))
      ..add(session.connection.stream.listen((_) => _sessionChanged(session)))
      // A reply that fails ends with nothing written in the channel: only the socket says so.
      ..add(session.events().listen((event) => _report(event, session)));
    _sessionChanged(session);
  }

  /// Reads the window of thinking [message] carries, and says whether it added any.
  bool _readReasoning(stream.Message message) {
    if (message.extraData['reasoning'] case {'id': final String step}) {
      final key = (message: message.id, step: step);
      final before = reasoning[key];
      reasoning.read(message);
      return !identical(reasoning[key], before);
    }
    return false;
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
    if (liveUI.observe(session.conversation.value)) {
      _publish();
      return;
    }
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
    // A card's tool still being added is waited for, so the reply can call it.
    await _adding;
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
      await _adding;
      if (_closed) return;
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

  /// Gives the agent a tool a card in the conversation brings, unless it has it already. A card
  /// is shown in a reply, so the conversation has its session by then. One that could not be
  /// added is added again the next time a card brings it.
  Future<void> bring(AgentTool tool) {
    final session = _session;
    if (session == null || _closed || !_brought.add(tool.name)) return _adding ?? Future.value();
    return _adding = _add(session, tool);
  }

  Future<void> _add(AgentSession session, AgentTool tool) async {
    try {
      await session.addTools([tool]);
    } on Object catch (error) {
      _brought.remove(tool.name);
      if (!_closed) _failure.value = '$error';
    }
  }

  /// Takes back a tool a card brought, once the cards have done what they were for. One that
  /// could not be taken back is taken back the next time this is asked. A question, and a card
  /// shown meanwhile, wait for it.
  Future<void> withdraw(String name) {
    final session = _session;
    if (session == null || _closed || !_brought.contains(name)) return Future.value();
    return _adding = _withdraw(session, name);
  }

  Future<void> _withdraw(AgentSession session, String name) async {
    try {
      await session.removeTool(name);
      _brought.remove(name);
    } on Object catch (error) {
      if (!_closed) _failure.value = '$error';
      return;
    }
    // A card shown while the tool was taken back still offers tables to book.
    if (_offers.keys.any((offer) => !_reservations.value.containsKey(offer))) {
      await bring(SeatSelectorComponent.reserveTable(reserve));
    }
  }

  /// The offer the person last tapped each table on, by the table's id, which a booking of that
  /// table comes from when more than one offer has it.
  final _chosen = <String, String>{};

  /// Notes that the person tapped table [tableId] on the seat selector of reply [offer].
  void choose(String offer, String tableId) => _chosen[tableId] = offer;

  /// Offers [tables] from the seat selector of reply [offer], or its newer version, and gives
  /// the agent the tool to book them while the offer is not booked.
  void offer(String offer, List<DiningTableItem> tables) {
    if (_closed) return;
    _offers[offer] = tables;
    if (_reservations.value.containsKey(offer)) return;
    unawaited(bring(SeatSelectorComponent.reserveTable(reserve)));
  }

  /// Books table [tableId] for [when] from the newest offer that has it free and is not booked,
  /// and takes the tool back once every offer is booked. The tool's answer does not wait for
  /// that, since the router takes the tool back only once the call that booked has finished.
  ///
  /// Throws a [ToolFailure] naming the free tables when no offer has it.
  Reservation reserve(String tableId, String when) {
    if (_closed) throw const ToolFailure('The conversation has closed.');
    final booked = _reservations.value;
    final newestFirst = _offers.entries.toList().reversed;
    final chosen = _chosen[tableId];
    // The offer the table was tapped on is the one booked: if it can no longer be, the table is
    // refused rather than booked at another venue whose table shares the id.
    final candidates = switch (_offers[chosen]) {
      final tables? => [MapEntry(chosen!, tables)],
      null => newestFirst,
    };
    for (final MapEntry(key: offer, value: tables) in candidates) {
      if (booked.containsKey(offer)) continue;
      final table = tables.where((t) => t.id == tableId && t.available).firstOrNull;
      if (table == null) continue;
      final reservation = (table: table, when: when, code: _confirmationCode(), offer: offer);
      _reservations.value = {...booked, offer: reservation};
      _chosen.remove(tableId);
      if (_offers.keys.every(_reservations.value.containsKey)) {
        unawaited(withdraw(SeatSelectorComponent.reserveTableTool));
      }
      return reservation;
    }
    final free = [
      for (final MapEntry(key: offer, value: tables) in _offers.entries)
        if (!booked.containsKey(offer))
          for (final table in tables)
            if (table.available) table.id,
    ];
    throw ToolFailure(
      '$tableId is not a free table on a seat selector. The free ones are ${free.join(', ')}.',
    );
  }

  static String _confirmationCode() {
    final random = math.Random.secure();
    return [for (var i = 0; i < 6; i++) '0123456789ABCDEF'[random.nextInt(16)]].join();
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
    _reservations.dispose();
    _usedSessionId.dispose();
    _sessionFailure.dispose();
    _failure.dispose();
  }
}

/// Makes [chat] the conversation the cards below it are shown in, which a card that brings its
/// own tool, such as the seat selector, gives it to.
class AgentChatScope extends InheritedWidget {
  const AgentChatScope({super.key, required this.chat, required super.child});

  final AgentChat chat;

  /// The conversation above [context], or null where a card is shown outside one.
  static AgentChat? maybeOf(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<AgentChatScope>()?.chat;

  @override
  bool updateShouldNotify(AgentChatScope oldWidget) => oldWidget.chat != chat;
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
  late final AIUIDocument? uiDocument = AIUIDocument.fromCustomData(message.extraData);

  @override
  late final List<AIMessageAttachment> attachments = [
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
