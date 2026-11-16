import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_demo/agent_chat.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_agents/stream_agents.dart';

void main() {
  test('describeWork names each skill in words', () {
    expect(
      describeWork(['refund_decision', 'order_status']),
      'Working on refund decision and order status',
    );
  });

  test('describeWork says nothing when no skill is named', () {
    expect(describeWork(const []), isNull);
  });

  group('AgentChat.activityOf', () {
    test('is thinking while responding or working with no named skill', () {
      const thinking = AIAgentActivity(state: AIAgentActivityState.thinking);
      expect(AgentChat.activityOf(ConversationState.responding), thinking);
      expect(AgentChat.activityOf(const Working([])), thinking);
    });

    test('names the skills being worked on', () {
      expect(
        AgentChat.activityOf(const Working(['refund_decision'])),
        const AIAgentActivity(
          state: AIAgentActivityState.checkingExternalSources,
          text: 'Working on refund decision',
        ),
      );
    });

    test('is null when idle or ended', () {
      expect(AgentChat.activityOf(ConversationState.idle), isNull);
      expect(AgentChat.activityOf(ConversationState.ended), isNull);
    });
  });

  test(
    'AgentChat.listed leaves out deleted messages and unwritten replies',
    () {
      final messages = [
        _question('q1'),
        stream.Message(
          id: 'gone',
          text: 'Deleted',
          deletedAt: DateTime.utc(2026),
        ),
        _reply('empty', generating: false),
        _reply('starting', generating: true),
        _reply('done', text: 'It ships on Friday.'),
      ];

      final listed = AgentChat.listed(messages, LiveReasoning());

      expect([for (final m in listed) m.id], ['q1', 'starting', 'done']);
    },
  );

  test('AgentReply.steps reads the ai_ attachments in order, by their ids', () {
    final reply = _reply(
      'a1',
      attachments: [
        stream.Attachment(
          type: 'ai_reasoning',
          id: 'r1',
          extraData: const {'v': 1, 'status': 'completed'},
        ),
        stream.Attachment(type: 'image', imageUrl: 'https://example.com/a.png'),
        _toolCall('call_1'),
      ],
    );

    expect([for (final step in reply.steps) step.id], ['r1', 'call_1']);
  });

  test(
    'AgentMessage.attachments shows images and files and leaves out steps',
    () {
      final message = AgentMessage(
        _question(
          'q1',
          attachments: [
            stream.Attachment(
              type: 'image',
              imageUrl: 'https://example.com/a.png',
              title: 'a.png',
            ),
            stream.Attachment(
              type: 'file',
              assetUrl: 'https://example.com/b.pdf',
              title: 'b.pdf',
              extraData: const {'mime_type': 'application/pdf'},
            ),
            _toolCall('call_1'),
            stream.Attachment(type: 'image'),
          ],
        ),
        LiveReasoning(),
      );

      expect(message.attachments, [
        AIMessageAttachment.image(
          Uri.parse('https://example.com/a.png'),
          name: 'a.png',
        ),
        AIMessageAttachment.file(
          Uri.parse('https://example.com/b.pdf'),
          name: 'b.pdf',
          mimeType: 'application/pdf',
        ),
      ]);
    },
  );

  test(
    "AgentMessage.role is the assistant's for a reply and the user's otherwise",
    () {
      final reasoning = LiveReasoning();

      expect(
        AgentMessage(_reply('a1'), reasoning).role,
        AIMessageRole.assistant,
      );
      expect(AgentMessage(_question('q1'), reasoning).role, AIMessageRole.user);
    },
  );

  group('AgentChat', () {
    late _Router router;
    late StreamAgents agents;
    late stream.StreamChatClient client;
    late AgentSession session;

    setUp(() async {
      router = await _Router.start();
      agents = StreamAgents(url: router.url, customerId: 'acme', userId: 'ada');
      client = stream.StreamChatClient('stream-key');
      session = await agents.attach('s1');
    });

    tearDown(() async {
      await session.close();
      await client.dispose();
      agents.close();
      await router.close();
    });

    stream.Channel channelOf([List<stream.Message> messages = const []]) {
      final channel = client.channel('messaging', id: 'agent-s1');
      channel.state = stream.ChannelClientState(
        channel,
        stream.ChannelState(
          channel: stream.ChannelModel(type: 'messaging', id: 'agent-s1'),
          messages: messages,
        ),
      );
      return channel;
    }

    void receive(stream.Channel channel, stream.Message message) =>
        client.handleEvent(
          stream.Event(
            type: stream.EventType.messageNew,
            cid: channel.cid,
            message: message,
          ),
        );

    void update(stream.Channel channel, stream.Message message) =>
        client.handleEvent(
          stream.Event(
            type: stream.EventType.messageUpdated,
            cid: channel.cid,
            message: message,
          ),
        );

    AgentChat closing(AgentChat chat) {
      addTearDown(() async {
        await chat.close();
        chat.dispose();
      });
      return chat;
    }

    AIConversation<AgentMessage> conversationOf(AgentChat chat) {
      final conversation = AIConversation(transport: chat);
      addTearDown(conversation.dispose);
      return conversation;
    }

    test(
      'publishes messages, skills activity, and llm failure with retry',
      () async {
        final channel = channelOf([_question('q0'), _reply('r0', text: 'Hi.')]);
        final chat = closing(
          AgentChat.following(session, channel, hasOlderMessages: true),
        );
        final conversation = conversationOf(chat);
        final socket = await router.socket;

        expect([for (final m in conversation.messages) m.id], ['q0', 'r0']);
        expect(conversation.hasOlderMessages, isTrue);

        conversation.send('Where is order A-1042?');
        await _until(() => router.asked == 1);
        receive(channel, _question('q1'));
        receive(channel, _reply('r1', generating: true));

        socket.send('{"type":"responding","turn_id":"t1"}');
        await _until(() => conversation.activity != null);
        expect(
          conversation.activity,
          const AIAgentActivity(state: AIAgentActivityState.thinking),
        );

        socket.send('{"type":"delegated","skill":"refund_decision"}');
        await _until(() => conversation.activity?.text != null);
        expect(conversation.activity?.text, 'Working on refund decision');

        socket.send('{"type":"error","context":"llm","error":"provider 500"}');
        await _until(() => conversation.error != null);
        update(channel, _reply('r1', generating: false));
        await pumpEventQueue();

        expect(
          conversation.error,
          isA<AIAgentError>().having(
            (e) => e.message,
            'message',
            AgentChat.unanswered,
          ),
        );
        expect(
          [for (final m in conversation.messages) m.id],
          ['q0', 'r0', 'q1'],
        );

        conversation.retry();
        await _until(() => router.asked == 2);
        expect(conversation.error, isNull);
      },
    );

    test('send failure surfaces on the conversation and retries', () async {
      router.responsesStatus = 500;
      final chat = closing(AgentChat.following(session, channelOf()));
      final conversation = conversationOf(chat);

      conversation.send('Where is my order?');
      await _until(() => conversation.error != null);
      expect(conversation.status, AIConversationStatus.error);

      router.responsesStatus = 202;
      conversation.retry();
      await _until(() => router.asked == 2);
      expect(conversation.error, isNull);
    });

    test('stop sends interrupt over the socket', () async {
      final chat = closing(AgentChat.following(session, channelOf()));
      final conversation = conversationOf(chat);

      conversation.send('Where is my order?');
      await _until(() => router.asked == 1);
      await chat.stop();

      expect((await (await router.socket).next())['type'], 'interrupt');
    });

    test('reuses unchanged AgentMessage instances across updates', () async {
      final channel = channelOf([
        _question('q0'),
        _reply('r0', text: 'Earlier.'),
      ]);
      final chat = closing(AgentChat.following(session, channel));
      final before = chat.listMessages;

      receive(channel, _question('q1'));
      await _until(() => chat.listMessages.length == 3);

      expect(chat.listMessages[0], same(before[0]));
      expect(chat.listMessages[1], same(before[1]));
    });

    test("reassembles a reply's live reasoning updates", () async {
      final channel = channelOf();
      final chat = closing(AgentChat.following(session, channel));
      receive(channel, _reply('r1', generating: true));

      for (final (offset, text) in [(0, 'Checking '), (9, 'policy.')]) {
        update(
          channel,
          stream.Message(
            id: 'r1',
            extraData: {
              'ai_generated': true,
              'generating': true,
              'reasoning': {'id': 'r1', 'offset': offset, 'text': text},
            },
          ),
        );
      }

      await _until(
        () => chat.reasoning[(message: 'r1', step: 'r1')] == 'Checking policy.',
      );
    });

    test('close disposes the watched channel', () async {
      final channel = channelOf([_question('q0')]);
      final chat = AgentChat.following(session, channel);

      await chat.close();
      chat.dispose();

      expect(channel.state, isNull);
    });
  });
}

stream.Message _question(
  String id, {
  List<stream.Attachment> attachments = const [],
}) => stream.Message(
  id: id,
  text: 'Where is order A-1042?',
  attachments: attachments,
);

stream.Message _reply(
  String id, {
  String text = '',
  bool generating = false,
  List<stream.Attachment> attachments = const [],
}) => stream.Message(
  id: id,
  text: text,
  attachments: attachments,
  extraData: {'ai_generated': true, 'generating': generating},
);

stream.Attachment _toolCall(String id) => stream.Attachment(
  type: 'ai_tool_call',
  id: id,
  extraData: const {'v': 1, 'status': 'running', 'name': 'lookup_order'},
);

/// A router on loopback that answers what a chat session does: reading the session, its
/// events socket, and asking, with [responsesStatus].
class _Router {
  _Router._(this._server) {
    _server.listen((request) async {
      if (WebSocketTransformer.isUpgradeRequest(request)) {
        _socket.complete(_Socket(await WebSocketTransformer.upgrade(request)));
        return;
      }
      await utf8.decodeStream(request);
      final asking = request.uri.path.endsWith('/responses');
      if (asking) asked += 1;
      request.response
        ..statusCode = asking ? responsesStatus : 200
        ..headers.contentType = ContentType.json
        ..write(jsonEncode(asking ? _answer(responsesStatus) : _session));
      await request.response.close();
    });
  }

  static Future<_Router> start() async =>
      _Router._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  final HttpServer _server;
  final _socket = Completer<_Socket>();

  /// The events socket, once the session has opened it.
  Future<_Socket> get socket => _socket.future;

  /// The status asking is answered with.
  int responsesStatus = 202;

  /// How many times `/responses` was called.
  int asked = 0;

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  Future<void> close() async {
    if (_socket.isCompleted) await (await _socket.future).close();
    await _server.close(force: true);
  }

  static Map<String, Object?> _answer(int status) => status < 300
      ? {
          'id': 'r1',
          'session_id': 's1',
          'said': '',
          'status': 'in_progress',
          'created_at': '2026-09-24T15:54:56Z',
        }
      : {
          'error': {
            'message': 'something went wrong',
            'type': 'internal',
            'code': 'internal',
            'doc_url': 'https://getstream.io/agents/docs/api/errors/#internal',
          },
        };

  static const _session = {
    'agent': 'support',
    'agent_id': 'a1',
    'call_id': '',
    'call_type': 'agent',
    'config_id': 'c1',
    'conversation_id': '',
    'created_at': '2026-09-24T09:54:56.038055-06:00',
    'id': 's1',
    'modality': 'text',
    'mode': 'text',
    'state': 'live',
    'text': true,
    'user_id': 'ada',
  };
}

/// The session's events socket, from the router's end.
class _Socket {
  _Socket(this._socket) {
    _socket.listen((frame) {
      final decoded = jsonDecode('$frame') as Map<String, Object?>;
      _frames.add(decoded);
      // As the router does once a session is told to close, which the app waits for.
      if (decoded['type'] == 'close') {
        unawaited(_socket.close(1000, 'the session ended'));
      }
    });
  }

  final WebSocket _socket;
  final _frames = StreamController<Map<String, Object?>>();
  late final _next = StreamIterator(_frames.stream);

  void send(String frame) => _socket.add(frame);

  /// The next frame the session sent.
  Future<Map<String, Object?>> next() async {
    await _next.moveNext().timeout(const Duration(seconds: 5));
    return _next.current;
  }

  Future<void> close() => _socket.close();
}

Future<void> _until(bool Function() done) async {
  final deadline = DateTime.now().add(const Duration(seconds: 5));
  while (!done()) {
    if (DateTime.now().isAfter(deadline)) fail('gave up waiting');
    await Future<void>.delayed(const Duration(milliseconds: 10));
  }
}
