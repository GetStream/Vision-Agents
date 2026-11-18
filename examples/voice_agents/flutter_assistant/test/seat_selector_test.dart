import 'dart:convert';
import 'dart:io' show HttpOverrides;

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_assistant/agent_chat.dart';
import 'package:flutter_assistant/brand_theme.dart';
import 'package:flutter_assistant/openui_renderer.dart';
import 'package:flutter_assistant/trip_components.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'support/fake_router.dart';
import 'support/fake_sessions.dart';
import 'support/ui_message.dart';

void main() {
  group('SeatSelectorComponent', () {
    late FakeSessions sessions;

    setUp(() async {
      // The test binding answers every request with a 400; this test talks to a router on
      // this machine.
      HttpOverrides.global = null;
      sessions = await FakeSessions.onRouter();
    });
    tearDown(() => sessions.dispose());

    /// Has the router call `reserve_table` for [tableId], allows it as the person would, and
    /// returns the result sent back.
    Future<Map<String, Object?>> call(AgentChat chat, String tableId) async {
      (await sessions.socket()).add(
        jsonEncode({
          'type': 'tool_call',
          'id': 'c1',
          'name': SeatSelectorComponent.reserveTableTool,
          'arguments': jsonEncode({
            'table_id': tableId,
            'when': 'tonight at 8pm',
            'summary': 'Harbor Window, for 4, tonight at 8pm',
          }),
        }),
      );
      final session = chat.session!;
      await until(() => session.approvals.value.isNotEmpty);
      session.decide('c1', allowed: true);
      await until(() => sessions.said.any((m) => m.contains('"tool_result"')));
      return jsonDecode(sessions.said.lastWhere((m) => m.contains('"tool_result"')))
          as Map<String, Object?>;
    }

    testWidgets('offers every table of a card written in pieces', (tester) async {
      final chat = AgentChat('s1', sessions: sessions);
      addTearDown(chat.dispose);
      addTearDown(() => tester.runAsync(chat.close));
      await tester.runAsync(() async {
        await TripComponents.load();
        await chat.open();
      });
      final host = AIUIHost(
        components: TripComponents.all,
        renderers: [OpenUIRenderer()],
        send: (_) async {},
      );
      addTearDown(host.dispose);
      Future<void> show(UIMessage reply) async {
        host.receive([reply]);
        await tester.pumpWidget(
          MaterialApp(
            theme: ThemeData(extensions: [StreamTheme.light()]),
            home: Scaffold(
              body: AgentChatScope(
                chat: chat,
                child: BrandThemeScope(
                  child: AIUIView(host: host, document: reply.uiDocument, messageId: 'm1'),
                ),
              ),
            ),
          ),
        );
        await tester.pump();
      }

      // The agent has written the first table, and is still writing.
      await show(
        UIMessage(
          'm1',
          'root = SeatSelector([{id: "T1", zone: "Cliff Terrace", seats: 2, available: true}])',
          isStreaming: true,
        ),
      );
      await show(
        UIMessage(
          'm1',
          'root = SeatSelector([{id: "T1", zone: "Cliff Terrace", seats: 2, available: true}, '
              '{id: "T4", zone: "Harbor Window", seats: 4, available: true}])',
        ),
      );

      // The card asks for its tool from the widget tree, whose work runs as frames are pumped.
      for (
        var i = 0;
        i < 500 && !sessions.routes.contains('POST /v1/agents/sessions/s1/tools');
        i++
      ) {
        await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
        await tester.pump();
      }
      final result = await tester.runAsync(() => call(chat, 'T4'));

      expect(result?['output'], contains('Reserved table T4'));
      expect(chat.reservation.value?.tables, {'T1', 'T4'});
      await tester.pumpWidget(const SizedBox());
      // The HTTP client the card's request went out on lets its idle connection go after 15 s.
      await tester.pump(const Duration(seconds: 16));
    });
  });
}
