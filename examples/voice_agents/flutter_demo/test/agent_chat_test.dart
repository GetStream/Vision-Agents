import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_demo/agent_chat.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_agents_chat/stream_agents_chat.dart';
import 'package:stream_agents_core/stream_agents_core.dart';

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

  test(
    'isThinkingAbout is false when nothing was asked and the session is idle',
    () {
      expect(
        isThinkingAbout(
          state: ConversationState.idle,
          messages: [
            _question('q1'),
            _reply('a1', text: 'It is on its way.'),
          ],
          askedAfter: null,
        ),
        isFalse,
      );
    },
  );

  test('isThinkingAbout is true as soon as the first question is asked', () {
    expect(
      isThinkingAbout(
        state: ConversationState.idle,
        messages: const [],
        askedAfter: '',
      ),
      isTrue,
    );
  });

  test('isThinkingAbout is true before the question reaches the channel', () {
    expect(
      isThinkingAbout(
        state: ConversationState.idle,
        messages: [_reply('a1', text: 'Hello.')],
        askedAfter: 'a1',
      ),
      isTrue,
    );
  });

  test(
    'isThinkingAbout is true while the reply is generating with nothing to show',
    () {
      expect(
        isThinkingAbout(
          state: ConversationState.idle,
          messages: [_question('q1'), _reply('a1', generating: true)],
          askedAfter: '',
        ),
        isTrue,
      );
    },
  );

  test('isThinkingAbout is false once the reply shows text', () {
    expect(
      isThinkingAbout(
        state: ConversationState.responding,
        messages: [
          _question('q1'),
          _reply('a1', text: 'Your order', generating: true),
        ],
        askedAfter: '',
      ),
      isFalse,
    );
  });

  test('isThinkingAbout is false once the reply has finished', () {
    expect(
      isThinkingAbout(
        state: ConversationState.idle,
        messages: [
          _question('q1'),
          _reply('a1', text: 'It ships on Friday.'),
        ],
        askedAfter: '',
      ),
      isFalse,
    );
  });

  test('isThinkingAbout is false once the reply shows a step', () {
    expect(
      isThinkingAbout(
        state: ConversationState.responding,
        messages: [
          _question('q1'),
          _reply('a1', generating: true, attachments: [_toolCall('call_1')]),
        ],
        askedAfter: '',
      ),
      isFalse,
    );
  });

  test(
    'isThinkingAbout is false while skills say what they are working on',
    () {
      expect(
        isThinkingAbout(
          state: const Working(['refund_decision']),
          messages: [_question('q1')],
          askedAfter: '',
        ),
        isFalse,
      );
    },
  );

  test(
    'isThinkingAbout is true while the model responds to a question asked elsewhere',
    () {
      expect(
        isThinkingAbout(
          state: ConversationState.responding,
          messages: [_question('q1')],
          askedAfter: null,
        ),
        isTrue,
      );
    },
  );

  test(
    'AgentReply.isEmptyReply is true for a reply with no text and no steps',
    () {
      expect(_reply('a1', generating: true).isEmptyReply, isTrue);
    },
  );

  test("AgentReply.isEmptyReply is false for the person's own message", () {
    expect(_question('q1').isEmptyReply, isFalse);
  });

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
    late AgentSession session;

    setUp(() async {
      router = await _Router.start();
      agents = StreamAgents(url: router.url, customerId: 'acme', userId: 'ada');
      session = await agents.attach('s1');
    });

    tearDown(() async {
      await session.close();
      agents.close();
      await router.close();
    });

    test('ask keeps why the question could not be sent', () async {
      router.responsesStatus = 500;
      final chat = AgentChat.following(session, _channel([_question('q1')]));

      await chat.ask('Where is my order?');

      expect(chat.failure, isNotNull);
      expect(chat.isThinking, isFalse);
    });

    test(
      'ask clears the last failure and thinks until the reply begins',
      () async {
        router.responsesStatus = 500;
        final chat = AgentChat.following(session, _channel([_question('q1')]));
        await chat.ask('Where is my order?');
        router.responsesStatus = 202;

        await chat.ask('Where is my order?');

        expect(chat.failure, isNull);
        expect(chat.isThinking, isTrue);
      },
    );

    test('stop stops thinking about the question asked', () async {
      final chat = AgentChat.following(session, _channel([_question('q1')]));
      await chat.ask('Where is my order?');

      chat.stop();

      expect(chat.isThinking, isFalse);
      expect((await (await router.socket).next())['type'], 'interrupt');
    });

    test('isAnswering while the last reply is still being written', () {
      final chat = AgentChat.following(
        session,
        _channel([
          _question('q1'),
          _reply('a1', text: 'It ships', generating: true),
        ]),
      );

      expect(chat.isAnswering, isTrue);
    });

    test('isAnswering is false once the last reply is written', () {
      final chat = AgentChat.following(
        session,
        _channel([_question('q1'), _reply('a1', text: 'It ships on Friday.')]),
      );

      expect(chat.isAnswering, isFalse);
    });

    test(
      'isAnswering while the session is responding, before any reply',
      () async {
        final chat = AgentChat.following(session, _channel([_question('q1')]));

        (await router.socket).send('{"type":"responding","turn_id":"t1"}');
        await _until(() => chat.session.state is Responding);

        expect(chat.isAnswering, isTrue);
      },
    );
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

/// The channel [messages] are in, as watching it would have opened it.
stream.Channel _channel(List<stream.Message> messages) =>
    stream.Channel.fromState(
      stream.StreamChatClient('stream-key'),
      stream.ChannelState(
        channel: stream.ChannelModel(type: 'messaging', id: 'agent-s1'),
        messages: messages,
      ),
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
    _socket.listen(
      (frame) => _frames.add(jsonDecode('$frame') as Map<String, Object?>),
    );
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
