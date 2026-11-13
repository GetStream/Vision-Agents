import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_agents/stream_agents.dart';

/// Connects Stream's [AIConversation] to a written conversation's session and channel.
///
/// The router writes everything here. Asking goes through the session, which records the
/// question in the channel and answers it there: the reply's text, and its steps (each round
/// of reasoning and each tool call) as attachments. The session's socket is what runs this
/// phone's tools and carries the person's answers to the ones that ask first.
class AgentChat with AIConversationTransport<AgentMessage> {
  AgentChat._(this.session, stream.Channel channel) : _channel = channel;

  /// Opens the session's channel and starts following it.
  ///
  /// The session is closed if its channel cannot be opened, so it does not run on with
  /// nobody to show it.
  static Future<AgentChat> open(AgentSession session) async {
    stream.Channel? channel;
    try {
      channel = await session.chat();
      final watched = await channel.watch(
        messagesPagination: const stream.PaginationParams(limit: _pageSize),
      );
      return AgentChat._(session, channel)
        .._hasOlderMessages = (watched.messages?.length ?? 0) >= _pageSize
        .._follow(channel);
    } on Object {
      channel?.dispose();
      await session.close();
      rethrow;
    }
  }

  /// Follows [channel], already open, for [session].
  @visibleForTesting
  factory AgentChat.following(
    AgentSession session,
    stream.Channel channel, {
    bool hasOlderMessages = false,
  }) => AgentChat._(session, channel)
    .._hasOlderMessages = hasOlderMessages
    .._follow(channel);

  static const _pageSize = 30;

  /// What a reply that failed says, in place of the provider's raw error.
  static const unanswered = "The agent couldn't answer.";

  final AgentSession session;
  stream.Channel? _channel;

  /// Why the session's socket stopped, or null: tools and approvals stop until a new
  /// conversation is opened.
  ValueListenable<String?> get sessionFailure => _sessionFailure;
  final _sessionFailure = ValueNotifier<String?>(null);

  /// Each reply's thinking in full, from its live updates. A stored step only keeps its
  /// opening.
  final reasoning = LiveReasoning();

  AIConversationFeed<AgentMessage>? _feed;
  final _subscriptions = <StreamSubscription<Object?>>[];
  final _shown = Expando<AgentMessage>();
  var _hasOlderMessages = false;
  AIAgentActivity? _activity;
  var _unanswered = false;
  var _closed = false;

  /// The messages as Stream's AI components show them.
  List<AgentMessage> get listMessages =>
      listed(_channel?.state?.messages ?? const [], reasoning, shown: _shown);

  /// The [messages] as Stream's AI components show them, oldest first, without those deleted
  /// and without the replies that ended with nothing written, reusing those in [shown].
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
    _tell(_unanswered ? _failedActivity : activityOf(session.state));
  }

  void _follow(stream.Channel channel) {
    _subscriptions
      ..add(channel.state!.messagesStream.listen((_) => _publish()))
      // The client's events rather than the channel's list: the list can fold several live
      // updates into one, and each update carries its own window of the reply's thinking.
      ..add(
        channel.client.on(stream.EventType.messageUpdated).listen((event) {
          final message = event.message;
          if (event.cid == channel.cid &&
              message != null &&
              _readReasoning(message)) {
            _publish();
          }
        }),
      )
      ..add(session.conversation.stream.listen((_) => _sessionChanged()))
      ..add(session.connection.stream.listen((_) => _sessionChanged()))
      // A reply that fails ends with nothing written in the channel: only the socket says so.
      ..add(session.events().listen(_report));
    _sessionChanged();
  }

  bool _readReasoning(stream.Message message) {
    if (message.extraData['reasoning'] case {'id': final String step}) {
      final key = (message: message.id, step: step);
      final before = reasoning[key];
      reasoning.read(message);
      return !identical(reasoning[key], before);
    }
    return false;
  }

  void _publish() {
    final feed = _feed;
    if (feed == null || _channel == null) return;
    feed
      ..updateMessages(listMessages)
      ..updateHasOlderMessages(_hasOlderMessages);
  }

  void _sessionChanged() {
    _sessionFailure.value = session.failure?.toString();
    if (!_unanswered) _tell(activityOf(session.state));
  }

  void _report(AgentEvent event) {
    if (failedReply(event) case final failed?) {
      _unanswered = true;
      _activity = failed;
      _feed?.updateActivity(failed);
    } else if (event.kind == AgentEventKind.responding) {
      _unanswered = false;
      _sessionChanged();
    }
  }

  void _tell(AIAgentActivity? activity) {
    if (activity == _activity) return;
    _activity = activity;
    _feed?.updateActivity(activity);
  }

  static const _failedActivity = AIAgentActivity(
    state: AIAgentActivityState.error,
    text: unanswered,
  );

  /// What the agent is doing in [state]: thinking while it answers, or working on its skills.
  static AIAgentActivity? activityOf(ConversationState state) =>
      switch (state) {
        Responding() => const AIAgentActivity(
          state: AIAgentActivityState.thinking,
        ),
        Working(:final skills) when skills.isEmpty => const AIAgentActivity(
          state: AIAgentActivityState.thinking,
        ),
        Working(:final skills) => AIAgentActivity(
          state: AIAgentActivityState.checkingExternalSources,
          text: describeWork(skills),
        ),
        _ => null,
      };

  /// What the agent was doing in [event] when it could not answer, or null otherwise.
  static AIAgentActivity? failedReply(AgentEvent event) =>
      event.kind == AgentEventKind.error && event['context'] == 'llm'
      ? _failedActivity
      : null;

  @override
  Future<void> send(AIConversationRequest<AgentMessage> request) async {
    if (_closed) throw const AIConversationCancelledException();
    await session.responses.create(request.text);
  }

  @override
  Future<void> stop() async {
    try {
      session.interrupt();
    } on SocketClosedException {
      // Nothing is answering: the reply in flight stopped with the socket.
    }
  }

  @override
  Future<void> loadOlderMessages() async {
    final channel = _channel;
    final first = channel?.state?.messages.firstOrNull;
    if (channel == null || first == null) return;
    final state = await channel.query(
      messagesPagination: stream.PaginationParams(
        lessThan: first.id,
        limit: _pageSize,
      ),
    );
    if (_closed) return;
    _hasOlderMessages = (state.messages?.length ?? 0) >= _pageSize;
    _publish();
  }

  /// Stops following the channel and closes the session.
  Future<void> close() async {
    _closed = true;
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    _channel?.dispose();
    _channel = null;
    await session.close();
  }

  /// Releases what the chat holds, once [close] has completed.
  void dispose() {
    _sessionFailure.dispose();
  }
}

/// What the agent says it is doing while its [skills] work, or null when it names none.
String? describeWork(List<String> skills) {
  if (skills.isEmpty) return null;
  return 'Working on ${skills.map((skill) => skill.replaceAll('_', ' ')).join(' and ')}';
}

extension AgentReply on stream.Message {
  /// Whether the agent wrote this message, which Stream's AI components render as a reply.
  bool get isAgentReply => extraData['ai_generated'] == true;

  /// Whether the agent is still writing it.
  bool get isGenerating => extraData['generating'] == true;

  /// Whether this is a reply the agent ended without writing anything: no text and no steps.
  bool get isUnwrittenReply =>
      isAgentReply &&
      !isGenerating &&
      (text ?? '').isEmpty &&
      attachments.isEmpty;

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
  AIMessageRole get role =>
      message.isAgentReply ? AIMessageRole.assistant : AIMessageRole.user;

  @override
  String get text => message.text ?? '';

  @override
  bool get isGenerating => message.isGenerating;

  @override
  late final List<AIMessageAttachment> attachments = [
    for (final attachment in message.attachments)
      if (attachment.type == 'image' && attachment.imageUrl != null)
        AIMessageAttachment.image(
          Uri.parse(attachment.imageUrl!),
          name: attachment.title,
        )
      else if (attachment.type == 'file' && attachment.assetUrl != null)
        AIMessageAttachment.file(
          Uri.parse(attachment.assetUrl!),
          name: attachment.title,
          size: attachment.fileSize,
          mimeType: attachment.mimeType,
        ),
  ];

  /// A step keeps only the opening of its thinking; the whole of it came with the live
  /// updates, while the reply was being written.
  @override
  String? reasoningText(AIReasoningPart step) =>
      _reasoning[(message: message.id, step: step.id)];
}
