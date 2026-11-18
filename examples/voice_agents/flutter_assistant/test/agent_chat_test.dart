import 'dart:async';
import 'dart:convert';

import 'package:flutter_assistant/agent_chat.dart';
import 'package:flutter_assistant/trip_components.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_agents/stream_agents.dart';

import 'support/fake_router.dart';
import 'support/fake_sessions.dart';

void main() {
  test('AgentChat.workingOnSkills names a skill in words', () {
    expect(AgentChat.workingOnSkills(['web_search']), 'Working on web search');
  });

  test('AgentChat.workingOnSkills names every skill being worked on', () {
    expect(AgentChat.workingOnSkills(['web_search', 'recall']), 'Working on web search and recall');
  });

  test('AgentReply.isAgentReply is true for a message the agent wrote', () {
    expect(_agentReply(text: 'Hello').isAgentReply, isTrue);
  });

  test("AgentReply.isAgentReply is false for the person's message", () {
    expect(stream.Message(id: 'q', text: 'Hi').isAgentReply, isFalse);
  });

  test('AgentReply.isGenerating is true while the agent is still writing', () {
    final reply = _agentReply(text: 'Hel', extraData: {'generating': true});

    expect(reply.isGenerating, isTrue);
  });

  group('AgentChat.activityOf', () {
    test('is thinking while the agent answers', () {
      expect(
        AgentChat.activityOf(ConversationState.responding),
        const AIAgentActivity(state: AIAgentActivityState.thinking),
      );
    });

    test('is thinking while the agent works with no skill named', () {
      expect(
        AgentChat.activityOf(const Working([])),
        const AIAgentActivity(state: AIAgentActivityState.thinking),
      );
    });

    test('names the skills the agent is working on', () {
      expect(
        AgentChat.activityOf(const Working(['web_search'])),
        const AIAgentActivity(
          state: AIAgentActivityState.checkingExternalSources,
          text: 'Working on web search',
        ),
      );
    });

    test('is nothing while the agent is idle or listening', () {
      expect(AgentChat.activityOf(ConversationState.idle), isNull);
      expect(AgentChat.activityOf(ConversationState.listening), isNull);
    });
  });

  group('AgentChat.failedReply', () {
    test("is an error saying Juno couldn't answer for a failed reply", () {
      final activity = AgentChat.failedReply(const AgentEvent('error', {'context': 'llm'}));

      expect(
        activity,
        const AIAgentActivity(state: AIAgentActivityState.error, text: AgentChat.unanswered),
      );
    });

    test('is nothing for an error of another part of the agent', () {
      expect(AgentChat.failedReply(const AgentEvent('error', {'context': 'tts'})), isNull);
    });

    test('is nothing for an event that is not an error', () {
      expect(AgentChat.failedReply(const AgentEvent('responding', {'context': 'llm'})), isNull);
    });
  });

  group('AgentChat.reasoningEffortOf', () {
    test('is high for Think harder', () {
      final option = ChatOption(id: AgentChat.thinkHarder, text: 'Think harder');

      expect(AgentChat.reasoningEffortOf(option), 'high');
    });

    test('is none for another option or for none', () {
      final option = ChatOption(id: 'other', text: 'Other');

      expect(AgentChat.reasoningEffortOf(option), isNull);
      expect(AgentChat.reasoningEffortOf(null), isNull);
    });
  });

  test('AgentChat.listed leaves out the messages that were deleted', () {
    final messages = [
      stream.Message(id: 'a', text: 'Hi'),
      stream.Message(id: 'b', text: 'Gone', deletedAt: DateTime.utc(2026)),
      _agentReply(text: 'Hello'),
    ];

    final listed = AgentChat.listed(messages, LiveReasoning());

    expect(listed.map((message) => message.id), ['a', 'r']);
  });

  test('AgentChat.listed leaves out a reply that ended with nothing written', () {
    final messages = [
      stream.Message(id: 'a', text: 'Hi'),
      _agentReply(id: 'failed', extraData: {'generating': false}),
      stream.Message(id: 'b', text: 'Again'),
      _agentReply(id: 'starting', extraData: {'generating': true}),
    ];

    final listed = AgentChat.listed(messages, LiveReasoning());

    expect(listed.map((message) => message.id), ['a', 'b', 'starting']);
  });

  group('AgentChat', () {
    late FakeSessions sessions;

    setUp(() async => sessions = await FakeSessions.onRouter());
    tearDown(() => sessions.dispose());

    /// Closes [chat] once the test is over, before the router goes.
    AgentChat closing(AgentChat chat) {
      addTearDown(() async {
        await chat.close();
        chat.dispose();
      });
      return chat;
    }

    AIConversation<AgentMessage> conversationOf(
      AgentChat chat, {
      ChatComposerController? composer,
    }) {
      final conversation = AIConversation(transport: chat, composer: composer);
      addTearDown(conversation.dispose);
      return conversation;
    }

    test(
      'tells the conversation its messages, older ones, what the agent does and its failure',
      () async {
        sessions.history = [
          for (var i = 0; i < 15; i++) ...[
            _question('q$i'),
            _agentReply(id: 'r$i', text: 'Reply $i'),
          ],
        ];
        final chat = closing(AgentChat('s1', sessions: sessions));
        final conversation = conversationOf(chat);

        await chat.open();
        expect(
          conversation.messages.map((message) => message.id),
          containsAllInOrder(['q0', 'r14']),
        );
        expect(conversation.messages, hasLength(30));
        expect(conversation.hasOlderMessages, isTrue);

        sessions.receive(_question('q15'));
        await until(() => conversation.messages.length == 31);
        expect(conversation.messages.last.id, 'q15');

        final socket = await sessions.socket();
        socket.add('{"type":"responding","turn_id":"t1"}');
        await until(() => conversation.activity != null);
        expect(conversation.activity, const AIAgentActivity(state: AIAgentActivityState.thinking));
        expect(conversation.status, AIConversationStatus.submitted);

        socket.add('{"type":"delegated","skill":"web_search"}');
        await until(() => conversation.activity?.text != null);
        expect(conversation.activity?.text, 'Working on web search');

        socket.add('{"type":"error","context":"llm","error":"402: see https://provider/billing"}');
        await until(() => conversation.error != null);
        expect(
          conversation.error,
          isA<AIAgentError>().having((error) => error.message, 'message', AgentChat.unanswered),
        );
        expect(conversation.status, AIConversationStatus.error);
      },
    );

    test(
      'a reply that failed stays failed, and can be asked again, as the router goes on',
      () async {
        sessions.history = [_question('q0'), _agentReply(id: 'r0', text: 'Earlier')];
        final chat = closing(AgentChat('s1', sessions: sessions));
        final conversation = conversationOf(chat);
        await chat.open();
        final socket = await sessions.socket();

        // The router records the question and opens its reply, empty, as it takes the question.
        conversation.send('And now?');
        await until(() => sessions.asked == 1);
        sessions
          ..receive(_question('q1'))
          ..receive(_agentReply(id: 'r1', extraData: {'generating': true}));
        socket.add('{"type":"responding","turn_id":"t1"}');
        await until(() => conversation.activity != null);

        socket.add('{"type":"error","context":"llm","error":"402 from provider"}');
        await until(() => conversation.error != null);
        // It then ends the reply with nothing written, and the session still says it responds.
        sessions.update(_agentReply(id: 'r1', extraData: {'generating': false}));
        socket.add('{"type":"error","context":"tts","error":"no voice"}');
        await until(() => chat.session!.conversation.value.failure == 'no voice');
        await pumpEventQueue();

        expect(
          conversation.error,
          isA<AIAgentError>().having((error) => error.message, 'message', AgentChat.unanswered),
        );
        expect(conversation.status, AIConversationStatus.error);
        expect(conversation.messages.map((message) => message.id), ['q0', 'r0', 'q1']);

        conversation.retry();
        await until(() => sessions.asked == 2);
        expect(conversation.error, isNull);
        expect(conversation.status, AIConversationStatus.submitted);

        // The next reply shows what the agent is doing again, and fails as the first did.
        socket.add('{"type":"responding","turn_id":"t2"}');
        await until(() => conversation.activity?.state == AIAgentActivityState.thinking);
        socket.add('{"type":"error","context":"llm","error":"402 from provider"}');
        await until(() => conversation.error != null);
        expect(conversation.status, AIConversationStatus.error);
      },
    );

    test('a conversation from the list takes questions once what was said in it is in', () async {
      sessions
        ..history = [_question('q0'), _agentReply(id: 'r0', text: 'Earlier')]
        ..hold = Completer();
      final chat = closing(AgentChat('s1', sessions: sessions));
      final conversation = conversationOf(chat);
      final opening = chat.open();

      expect(chat.takesQuestions, isFalse);

      sessions.hold!.complete();
      await opening;
      expect(chat.takesQuestions, isTrue);
      expect(conversation.messages, hasLength(2));

      // Asked now, the question waits for its own reply rather than taking the earlier one.
      conversation.send('And now?');
      await until(() => sessions.asked == 1);
      expect(conversation.pendingMessage?.text, 'And now?');
      expect(conversation.status, AIConversationStatus.submitted);
    });

    test('a new conversation takes questions at once', () {
      final chat = closing(AgentChat(null, sessions: sessions));

      expect(chat.takesQuestions, isTrue);
    });

    test(
      'a question with a photo that does not read goes back to the composer without it',
      () async {
        final thinkHarder = ChatOption(id: AgentChat.thinkHarder, text: 'Think harder');
        final composer = ChatComposerController(chatOptions: [thinkHarder]);
        addTearDown(composer.dispose);
        final chat = closing(AgentChat(null, sessions: sessions, composer: composer));
        final conversation = conversationOf(chat, composer: composer);

        conversation.send(
          'What is this?',
          attachments: [XFile('/no/such/photo.jpg')],
          chatOption: thinkHarder,
        );
        await until(() => chat.failure.value != null);

        expect(chat.failure.value, 'Could not read photo.jpg as a photo. Send again without it.');
        expect(composer.text, 'What is this?');
        expect(composer.attachments, isEmpty);
        expect(composer.selectedChatOption, thinkHarder);
        // The conversation forgets the question: nothing waits, and nothing is offered again.
        expect(conversation.pendingMessage, isNull);
        expect(conversation.status, AIConversationStatus.ready);
        expect(conversation.error, isNull);
        expect(sessions.asked, 0);
        expect(chat.usedSessionId.value, isNull);
      },
    );

    test('a question not asked stays out of a composer written in since', () async {
      final composer = ChatComposerController();
      addTearDown(composer.dispose);
      final chat = closing(AgentChat(null, sessions: sessions, composer: composer));
      final conversation = conversationOf(chat, composer: composer);

      conversation.send('What is this?', attachments: [XFile('/no/such/photo.jpg')]);
      composer.textEditingController.text = 'Something new';
      await until(() => chat.failure.value != null);

      expect(composer.text, 'Something new');
    });

    test('the next question clears why the last one could not be asked', () async {
      final chat = closing(AgentChat(null, sessions: sessions));
      final conversation = conversationOf(chat);
      conversation.send('What is this?', attachments: [XFile('/no/such/photo.jpg')]);
      await until(() => chat.failure.value != null);

      conversation.send('Never mind');

      await until(() => sessions.asked == 1);
      expect(chat.failure.value, isNull);
    });

    test('a question stopped while its session opens is not asked, and goes back', () async {
      sessions.hold = Completer();
      final composer = ChatComposerController();
      addTearDown(composer.dispose);
      final chat = closing(AgentChat(null, sessions: sessions, composer: composer));
      final conversation = conversationOf(chat, composer: composer);
      conversation.send('Plan a weekend');

      conversation.stop();
      sessions.hold!.complete();
      await until(() => conversation.pendingMessage == null);

      expect(composer.text, 'Plan a weekend');
      expect(conversation.status, AIConversationStatus.ready);
      expect(conversation.error, isNull);
      expect(sessions.asked, 0);
      expect(chat.usedSessionId.value, isNull);
    });

    test('a question stopped while the router takes it has its reply stopped', () async {
      final taken = Completer<void>();
      sessions.holdQuestion('s1', taken.future);
      final chat = closing(AgentChat(null, sessions: sessions));
      final conversation = conversationOf(chat);
      await chat.open();
      await until(() => chat.session!.isConnected);
      conversation.send('Plan a weekend');
      await until(() => sessions.asked == 1);

      conversation.stop();
      await until(() => sessions.interrupts == 1);
      taken.complete();

      // Stopping before the router had the question stopped nothing: it is told again.
      await until(() => sessions.interrupts == 2);
    });

    test('a reply the agent could not write is not the session failing', () async {
      final chat = closing(AgentChat('s1', sessions: sessions));
      await chat.open();

      (await sessions.socket()).add('{"type":"error","context":"llm","error":"402 from provider"}');
      await until(() => chat.session!.conversation.value.failure != null);

      expect(chat.sessionFailure.value, isNull);
    });

    test('sessionFailure says the socket dropped', () async {
      final chat = closing(AgentChat('s1', sessions: sessions));
      await chat.open();

      await (await sessions.socket()).close(1011, 'router restarting');
      await until(() => !chat.session!.isConnected);

      expect(chat.sessionFailure.value, contains('router restarting'));
    });

    test('stop does nothing once the socket has closed', () async {
      final chat = closing(AgentChat('s1', sessions: sessions));
      await chat.open();
      await (await sessions.socket()).close(1011);
      await until(() => !chat.session!.isConnected);

      await expectLater(chat.stop(), completes);
    });

    test('nothing opens, is used or is asked once closed', () async {
      final chat = AgentChat(null, sessions: sessions);
      await chat.close();
      final request = AIConversationRequest<AgentMessage>(
        kind: AIConversationRequestKind.message,
        text: 'Hi',
      );

      await expectLater(chat.open(), throwsA(isA<AIConversationCancelledException>()));
      await expectLater(chat.use(), throwsA(isA<AIConversationCancelledException>()));
      await expectLater(chat.send(request), throwsA(isA<AIConversationCancelledException>()));

      expect(sessions.started, 0);
      expect(chat.usedSessionId.value, isNull);
      chat.dispose();
    });

    test('a new conversation closed while it opens to be used is deleted, not used', () async {
      final chat = AgentChat(null, sessions: sessions);
      final used = expectLater(chat.use(), throwsA(isA<AIConversationCancelledException>()));

      await chat.close();

      await used;
      expect(chat.usedSessionId.value, isNull);
      expect(sessions.deleted, ['s1']);
      chat.dispose();
    });

    test('a new session whose channel does not open is deleted', () async {
      sessions.channelFailure = StateError('no channel');
      final chat = closing(AgentChat(null, sessions: sessions));

      await expectLater(chat.open(), throwsStateError);

      await until(() => sessions.deleted.isNotEmpty);
      expect(sessions.deleted, ['s1']);
      expect(chat.session, isNull);
    });

    test('opens again after an open that failed', () async {
      sessions.startFailure = StateError('offline');
      final chat = closing(AgentChat('s1', sessions: sessions));
      await expectLater(chat.open(), throwsStateError);
      sessions.startFailure = null;

      final session = await chat.open();

      expect(session.id, 's1');
      expect(sessions.started, 2);
      expect(chat.takesQuestions, isTrue);
    });

    test('close deletes a new conversation that was opened and never used', () async {
      final chat = AgentChat(null, sessions: sessions);
      await chat.open();

      await chat.close();

      expect(sessions.deleted, ['s1']);
      chat.dispose();
    });

    test('close keeps a new conversation that was used', () async {
      final chat = AgentChat(null, sessions: sessions);
      await chat.use();

      await chat.close();

      expect(sessions.deleted, isEmpty);
      expect(chat.usedSessionId.value, 's1');
      chat.dispose();
    });

    test('a message that did not change is shown as before when another arrives', () async {
      sessions.history = [_question('q0'), _agentReply(id: 'r0', text: 'Earlier')];
      final chat = closing(AgentChat('s1', sessions: sessions));
      await chat.open();
      final before = chat.listMessages;

      sessions.receive(_question('q1'));
      await until(() => chat.listMessages.length == 3);

      expect(chat.listMessages.first, same(before.first));
      expect(chat.listMessages[1], same(before[1]));
    });

    test("keeps a reply's thinking from its live updates", () async {
      final chat = closing(AgentChat('s1', sessions: sessions));
      await chat.open();
      sessions.receive(_agentReply(id: 'r1', extraData: {'generating': true}));

      for (final (offset, text) in [(0, 'Weighing '), (9, 'the options.')]) {
        sessions.update(
          _agentReply(
            id: 'r1',
            extraData: {
              'generating': true,
              'reasoning': {'id': 'r1', 'offset': offset, 'text': text},
            },
          ),
        );
      }

      await until(() => chat.reasoning[(message: 'r1', step: 'r1')] == 'Weighing the options.');
    });

    test('close lets go of the channel, before the session has closed', () async {
      final chat = AgentChat('s1', sessions: sessions);
      await chat.open();
      final channel = sessions.opened!;

      final closed = chat.close();

      expect(channel.state, isNull);
      await closed;
      chat.dispose();
    });

    test('close keeps a conversation from the list, used or not', () async {
      final chat = AgentChat('s1', sessions: sessions);
      await chat.open();

      await chat.close();

      expect(sessions.deleted, isEmpty);
      chat.dispose();
    });

    test('close waits for a session on its way, and deletes it if new and unused', () async {
      final chat = AgentChat(null, sessions: sessions);
      final opening = chat.open();

      await chat.close();

      await opening;
      expect(sessions.deleted, ['s1']);
      chat.dispose();
    });

    group('a seat selector', () {
      const tables = <DiningTableItem>[
        (id: 'T1', zone: 'Cliff Terrace', seats: 2, available: false),
        (id: 'T4', zone: 'Harbor Window', seats: 4, available: true),
      ];

      AgentTool reserveTable(AgentChat chat) =>
          SeatSelectorComponent.reserveTable(tables, chat.reserved);

      /// Has the router call `reserve_table` for [tableId], allows it as the person would, and
      /// returns the result sent back.
      Future<Map<String, Object?>> call(AgentChat chat, String tableId, {String id = 'c1'}) async {
        bool answered(String m) => m.contains('"tool_result"') && m.contains('"$id"');
        (await sessions.socket()).add(
          jsonEncode({
            'type': 'tool_call',
            'id': id,
            'name': SeatSelectorComponent.reserveTableTool,
            'arguments': jsonEncode({
              'table_id': tableId,
              'when': 'tonight at 8pm',
              'summary': 'Harbor Window, for 4, tonight at 8pm',
            }),
          }),
        );
        final session = chat.session!;
        await until(() => session.approvals.value.any((request) => request.call.id == id));
        session.decide(id, allowed: true);
        await until(() => sessions.said.any(answered));
        return jsonDecode(sessions.said.lastWhere(answered)) as Map<String, Object?>;
      }

      test('brings its tool to the session once, however often it shows', () async {
        final chat = closing(AgentChat('s1', sessions: sessions));
        await chat.open();

        await chat.bring(reserveTable(chat));
        await chat.bring(reserveTable(chat));

        expect(
          sessions.routes.where((r) => r == 'POST /v1/agents/sessions/s1/tools'),
          hasLength(1),
        );
      });

      test('reserves a free table, keeps the booking and takes its tool back', () async {
        final chat = closing(AgentChat('s1', sessions: sessions));
        await chat.open();
        await chat.bring(reserveTable(chat));

        final result = await call(chat, 'T4');

        expect(result['output'], contains('Reserved table T4'));
        expect(result['error'], isEmpty);
        expect(chat.reservation.value?.table.id, 'T4');
        expect(chat.reservation.value?.when, 'tonight at 8pm');
        expect(chat.reservation.value?.tables, {'T1', 'T4'});
        await until(
          () => sessions.routes.contains('DELETE /v1/agents/sessions/s1/tools/reserve_table'),
        );
      });

      test('refuses a taken table, naming the free ones, and keeps its tool', () async {
        final chat = closing(AgentChat('s1', sessions: sessions));
        await chat.open();
        await chat.bring(reserveTable(chat));

        final result = await call(chat, 'T1');

        expect(result['error'], allOf(contains('T1 is not a free table'), contains('T4')));
        expect(chat.reservation.value, isNull);
        // The tool is still there to reserve a free table.
        expect((await call(chat, 'T4', id: 'c2'))['output'], contains('Reserved table T4'));
      });

      test('a booking that arrives after the chat was let go of changes nothing', () async {
        final chat = AgentChat('s1', sessions: sessions);
        await chat.close();
        chat.dispose();

        chat.reserved((
          table: tables.last,
          when: 'tonight at 8pm',
          code: 'A1B2C3',
          tables: {'T1', 'T4'},
        ));
      });
    });

    test('opening twice at once starts one session', () async {
      final chat = closing(AgentChat(null, sessions: sessions));

      await Future.wait([chat.open(), chat.open()]);

      expect(sessions.started, 1);
    });
  });

  test("AgentMessage.role is the assistant's for the agent's reply", () {
    final message = AgentMessage(_agentReply(text: 'Hello'), LiveReasoning());

    expect(message.role, AIMessageRole.assistant);
  });

  test("AgentMessage.role is the user's for the person's message", () {
    final message = AgentMessage(stream.Message(id: 'q', text: 'Hi'), LiveReasoning());

    expect(message.role, AIMessageRole.user);
  });

  test('AgentMessage.attachments shows a photo by its URL', () {
    final question = stream.Message(
      id: 'q',
      attachments: [
        stream.Attachment(type: 'image', imageUrl: 'https://cdn.example.com/photo.jpg'),
      ],
    );

    final attachments = AgentMessage(question, LiveReasoning()).attachments;

    expect(attachments.single.url, Uri.parse('https://cdn.example.com/photo.jpg'));
  });

  test('AgentMessage.attachments leaves out a photo not uploaded yet', () {
    final question = stream.Message(
      id: 'q',
      attachments: [stream.Attachment(type: 'image')],
    );

    final attachments = AgentMessage(question, LiveReasoning()).attachments;

    expect(attachments, isEmpty);
  });
}

stream.Message _agentReply({
  String id = 'r',
  String? text,
  Map<String, Object?> extraData = const {},
}) => stream.Message(id: id, text: text, extraData: {'ai_generated': true, ...extraData});

stream.Message _question(String id) => stream.Message(id: id, text: 'Question $id');
