import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:vision_agents_chat/vision_agents_chat.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'photos.dart';

/// A written conversation as Stream's AI components want it: the channel it is kept in, the
/// whole of each reply's thinking, and what the agent is doing before it writes.
///
/// The router writes everything here. Asking goes through the session, which records the
/// question in the channel and answers it there: the reply's text, and its steps (each round
/// of reasoning and each tool call) as attachments. The session's socket is what runs this
/// phone's tools and carries the person's answers to the ones that ask first, on a call too.
class AgentChat extends ChangeNotifier {
  AgentChat._(this.session, this._channel);

  /// Opens the session's channel and starts following it.
  static Future<AgentChat> open(AgentSession session) async {
    final channel = await session.chat();
    final watched = await channel.watch(
      messagesPagination: const stream.PaginationParams(limit: _pageSize),
    );
    final chat = AgentChat._(session, channel).._listen();
    chat.hasOlderMessages = (watched.messages?.length ?? 0) >= _pageSize;
    return chat;
  }

  /// How many messages are loaded at a time.
  static const _pageSize = 30;

  final AgentSession session;
  final stream.Channel _channel;

  /// Each reply's thinking in full, from its live updates.
  ///
  /// A stored step only keeps its opening.
  final reasoning = LiveReasoning();

  /// How the person rated replies on this phone, by message.
  ///
  /// Kept for the session only: the router takes no ratings yet.
  // TODO: send the ratings once the router takes them.
  final ratings = <String, AIMessageRating>{};

  final _subscriptions = <StreamSubscription<Object?>>[];

  /// The channel's messages, oldest first.
  List<stream.Message> get messages => _channel.state?.messages ?? const [];

  /// Why asking failed, or null.
  String? failure;

  /// Whether the channel holds messages older than those loaded.
  bool hasOlderMessages = true;

  /// The newest message when something was last asked here, until it fails or is stopped:
  /// what arrives after it is the question and then its reply.
  String? _askedAfter;

  void _listen() {
    _subscriptions
      ..add(_channel.state!.messagesStream.listen((_) => notifyListeners()))
      // The client's events rather than the channel's list: the list can fold several live
      // updates into one, and each update carries its own window of the reply's thinking.
      ..add(
        _channel.client.on(stream.EventType.messageUpdated).listen((event) {
          if (event.cid == _channel.cid && event.message != null) {
            reasoning.read(event.message!);
            notifyListeners();
          }
        }),
      )
      ..add(session.conversation.stream.listen((_) => notifyListeners()))
      ..add(session.connection.stream.listen((_) => notifyListeners()));
  }

  /// The messages as Stream's AI components show them.
  List<AgentMessage> get listMessages => [
    for (final message in messages) AgentMessage(message, reasoning),
  ];

  /// What the agent is doing while its reply shows nothing yet: thinking, or working on its
  /// skills. Null otherwise.
  ///
  /// From the session's socket, which says what the agent is doing before anything of the
  /// reply reaches the channel.
  String? get workingOn => switch (session.state) {
    Working(:final skills) when skills.isNotEmpty => workingOnSkills(skills),
    _ => null,
  };

  /// What the agent is doing while it works on [skills], such as "Working on web search".
  static String workingOnSkills(List<String> skills) =>
      'Working on ${skills.map((skill) => skill.replaceAll('_', ' ')).join(' and ')}';

  /// Whether the agent is thinking about the latest question and its reply shows nothing.
  bool get isThinking {
    if (workingOn != null) return false;
    final last = messages.lastOrNull;
    final replyHasBegun = last != null && last.isAgentReply && !last.isEmptyReply;
    if (session.state is Responding && !replyHasBegun) return true;
    // The session can pass through idle between the question and the first of its reply.
    final askedAfter = _askedAfter;
    if (askedAfter == null || last == null || last.id == askedAfter) {
      return askedAfter != null;
    }
    return !last.isAgentReply || (last.isGenerating && !replyHasBegun);
  }

  /// Whether a reply is on its way, from the first thought to the last word.
  bool get isAnswering {
    if (session.state is Responding || session.state is Working) return true;
    return messages.lastWhereOrNull((m) => m.isAgentReply)?.isGenerating ?? false;
  }

  Future<void> loadOlderMessages() async {
    final first = messages.firstOrNull;
    if (first == null) return;
    final state = await _channel.query(
      messagesPagination: stream.PaginationParams(lessThan: first.id, limit: _pageSize),
    );
    hasOlderMessages = (state.messages?.length ?? 0) >= _pageSize;
    notifyListeners();
  }

  void rate(String messageId, AIMessageRating? rating) {
    if (rating == null) {
      ratings.remove(messageId);
    } else {
      ratings[messageId] = rating;
    }
    notifyListeners();
  }

  /// How many of the messages the person wrote, so a question on its way can tell when
  /// Stream's copy of it has arrived: it is the next one.
  int get personMessageCount => messages.where((m) => !m.isAgentReply).length;

  /// Asks, with [photos] from the composer and, when [thinkHarder], more time to reason.
  Future<void> ask(String text, {List<XFile> photos = const [], bool thinkHarder = false}) async {
    failure = null;
    _askedAfter = messages.lastOrNull?.id ?? '';
    notifyListeners();
    try {
      await session.responses.create(
        text,
        images: await Photos.sources(photos),
        reasoningEffort: thinkHarder ? 'high' : null,
      );
    } on Object catch (error) {
      failure = '$error';
      _askedAfter = null;
      notifyListeners();
    }
  }

  /// Stops the reply in flight. What was written so far is kept.
  void stop() {
    _askedAfter = null;
    try {
      session.interrupt();
    } on SocketClosedException {
      // Nothing is answering: the reply in flight stopped with the socket.
    }
    notifyListeners();
  }

  /// Stops following the channel and closes the session, which ends it on the router.
  ///
  /// Call [dispose] once it completes.
  Future<void> close() async {
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    await session.close();
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

  /// Whether it is a reply with nothing to show yet: no text and no steps.
  bool get isEmptyReply => isAgentReply && (text ?? '').isEmpty && steps.isEmpty;

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
