import 'dart:async';
import 'dart:convert';
import 'dart:io' show HttpOverrides;

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_assistant/agent_chat.dart';
import 'package:flutter_assistant/brand_theme.dart';
import 'package:flutter_assistant/trip_components.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'support/fake_router.dart';
import 'support/fake_sessions.dart';
import 'support/ui_message.dart';

const _tools = 'POST /v1/agents/sessions/s1/tools';
const _withdraw = 'DELETE /v1/agents/sessions/s1/tools/reserve_table';
const _asked = 'POST /v1/agents/sessions/s1/responses';

/// A seat selector card of [tables], each `id zone seats available`.
UIMessage _card(String id, List<(String, String, int, bool)> tables, {bool isStreaming = false}) {
  final items = [
    for (final (tableId, zone, seats, available) in tables)
      '{id: "$tableId", zone: "$zone", seats: $seats, available: $available}',
  ].join(', ');
  return UIMessage(id, 'root = SeatSelector([$items])', isStreaming: isStreaming);
}

void main() {
  group('SeatSelectorComponent', () {
    late FakeSessions sessions;
    late AgentChat chat;

    setUp(() async {
      // The test binding answers every request with a 400; these talk to a router on this
      // machine.
      final overrides = HttpOverrides.current;
      HttpOverrides.global = null;
      addTearDown(() => HttpOverrides.global = overrides);
      sessions = await FakeSessions.onRouter();
    });
    tearDown(() => sessions.dispose());

    /// Opens the conversation the cards are shown in.
    Future<void> open(WidgetTester tester) async {
      chat = AgentChat('s1', sessions: sessions);
      addTearDown(chat.dispose);
      addTearDown(() => tester.runAsync(chat.close));
      await tester.runAsync(() async {
        await TripComponents.load();
        await chat.open();
      });
    }

    /// Shows [replies] in the conversation, each with its card.
    Future<void> show(WidgetTester tester, List<UIMessage> replies) async {
      chat.ui.receive(replies);
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(extensions: [StreamTheme.light()]),
          home: Scaffold(
            body: AgentChatScope(
              chat: chat,
              child: BrandThemeScope(
                child: ListView(
                  children: [
                    for (final reply in replies)
                      AIUIView(host: chat.ui, document: reply.uiDocument, messageId: reply.id),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
      await tester.pump();
    }

    /// Pumps until [done], since a card's work runs as frames are pumped.
    Future<void> settle(WidgetTester tester, bool Function() done, String failure) async {
      final deadline = DateTime.now().add(const Duration(seconds: 5));
      while (!done()) {
        if (DateTime.now().isAfter(deadline)) fail(failure);
        await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
        await tester.pump();
      }
    }

    /// Has the router call `reserve_table` for [tableId] in [zone], allows it as the person
    /// would, and returns the result sent back.
    Future<Map<String, Object?>> call(
      WidgetTester tester,
      String tableId,
      String zone, {
      String id = 'c1',
      String? summary,
    }) async {
      bool answered(String m) => m.contains('"tool_result"') && m.contains('"$id"');
      final result = await tester.runAsync(() async {
        (await sessions.socket()).add(
          jsonEncode({
            'type': 'tool_call',
            'id': id,
            'name': SeatSelectorComponent.reserveTableTool,
            'arguments': jsonEncode({
              'table_id': tableId,
              'zone': zone,
              'when': 'tonight at 8pm',
              'summary': summary ?? '$tableId in $zone, for 2, tonight at 8pm',
            }),
          }),
        );
        final session = chat.session!;
        await until(() => session.approvals.value.any((request) => request.call.id == id));
        session.decide(id, allowed: true);
        await until(() => sessions.said.any(answered));
        return jsonDecode(sessions.said.lastWhere(answered)) as Map<String, Object?>;
      });
      await tester.pump();
      return result!;
    }

    /// Lets the HTTP client the cards' requests went out on drop its idle connection, which
    /// it does after 15 seconds.
    Future<void> finish(WidgetTester tester) async {
      await tester.pumpWidget(const SizedBox());
      await tester.pump(const Duration(seconds: 16));
    }

    bool brought() => sessions.routes.contains(_tools);

    testWidgets('offers every table of a card written in pieces', (tester) async {
      await open(tester);

      // The agent has written the first table, and is still writing.
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true)], isStreaming: true),
      ]);
      // A card still being written brings no tool: give a request time to arrive, and see none.
      for (var i = 0; i < 20; i++) {
        await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
        await tester.pump();
      }
      expect(sessions.routes, isNot(contains(_tools)));
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');

      final result = await call(tester, 'T4', 'Harbor Window');

      expect(result['output'], contains('Reserved table T4'));
      expect(find.text('Table T4 reserved'), findsOneWidget);
      await finish(tester);
    });

    testWidgets('books a table of a second card', (tester) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
        _card('m2', [('K3', 'Garden', 4, true)]),
      ]);
      await settle(tester, brought, 'The cards never brought their tool.');

      final result = await call(tester, 'K3', 'Garden');

      expect(result['output'], contains('Reserved table K3'));
      expect(find.text('Table K3 reserved'), findsOneWidget);
      await finish(tester);
    });

    testWidgets('keeps the first card\'s booking when a second card is booked', (tester) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
        _card('m2', [('K3', 'Garden', 4, true)]),
      ]);
      await settle(tester, brought, 'The cards never brought their tool.');

      await call(tester, 'T4', 'Harbor Window');
      final second = await call(tester, 'K3', 'Garden', id: 'c2');

      expect(second['output'], contains('Reserved table K3'));
      expect(find.text('Table T4 reserved'), findsOneWidget);
      expect(find.text('Table K3 reserved'), findsOneWidget);
      await finish(tester);
    });

    testWidgets('books the venue whose zone is named when two cards offer the same id', (
      tester,
    ) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Terrace', 2, true)]),
        _card('m2', [('T1', 'Garden', 2, true)]),
      ]);
      await settle(tester, brought, 'The cards never brought their tool.');

      await call(tester, 'T1', 'Terrace');

      // The older card is booked, and the newer one still offers its table.
      expect(find.text('Terrace · 2 people · tonight at 8pm'), findsOneWidget);
      expect(find.textContaining('T1 · Garden'), findsOneWidget);
      await finish(tester);
    });

    testWidgets('refuses a table taken since rather than booking another venue\'s of that id', (
      tester,
    ) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Terrace', 2, false)]),
        _card('m2', [('T1', 'Garden', 2, true)]),
      ]);
      await settle(tester, brought, 'The cards never brought their tool.');

      final result = await call(tester, 'T1', 'Terrace');

      expect(result['error'], contains('T1 in Terrace is not a free table'));
      expect(find.text('Garden · 2 people · tonight at 8pm'), findsNothing);
      await finish(tester);
    });

    testWidgets('books a table a finished card gained when it was written again', (tester) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Terrace', 2, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');
      await show(tester, [
        _card('m1', [('T1', 'Terrace', 2, false), ('T7', 'Garden', 4, true)]),
      ]);
      await tester.pump();

      // The table taken since is refused, and no card is booked by it.
      final taken = await call(tester, 'T1', 'Terrace', id: 'c0');
      expect(taken['error'], contains('T1 in Terrace is not a free table'));
      expect(find.text('Table T1 reserved'), findsNothing);
      final result = await call(tester, 'T7', 'Garden');

      expect(result['output'], contains('Reserved table T7'));
      await finish(tester);
    });

    testWidgets('refuses a second booking of a booked card while its tool is still there', (
      tester,
    ) async {
      await open(tester);
      sessions.answer(_withdraw, 500);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');
      await call(tester, 'T4', 'Harbor Window');
      await settle(
        tester,
        () => sessions.routes.contains(_withdraw),
        'The tool was never taken back.',
      );

      final second = await call(tester, 'T1', 'Cliff Terrace', id: 'c2');

      expect(second['error'], isNotEmpty);
      expect(find.text('Table T4 reserved'), findsOneWidget);
      await finish(tester);
    });

    testWidgets('asks about a table tapped while the card\'s tool is being added once it is', (
      tester,
    ) async {
      await open(tester);
      final added = Completer<void>();
      sessions.holdRoute(_tools, added.future);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');

      await tester.tap(find.textContaining('T4 · Harbor Window'));
      for (var i = 0; i < 20; i++) {
        await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
        await tester.pump();
      }
      expect(sessions.routes, isNot(contains(_asked)));

      added.complete();
      await settle(tester, () => sessions.routes.contains(_asked), 'The table was never asked.');
      await finish(tester);
    });

    testWidgets('adds the card\'s tool again when a table is tapped after adding it failed', (
      tester,
    ) async {
      await open(tester);
      sessions.answer(_tools, 500);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');
      await settle(tester, () => chat.failure.value != null, 'Adding the tool never failed.');
      sessions.answer(_tools, 204);

      await tester.tap(find.textContaining('T4 · Harbor Window'));
      await settle(
        tester,
        () => sessions.routes.where((route) => route == _tools).length == 2,
        'The tool was not added again.',
      );
      await settle(tester, () => sessions.routes.contains(_asked), 'The table was never asked.');
      await finish(tester);
    });

    testWidgets('refuses a booking whose summary names another table that starts the same', (
      tester,
    ) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T10', 'Garden', 4, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');

      final result = await call(
        tester,
        'T1',
        'Cliff Terrace',
        summary: 'T10 in the Garden, for 4, tonight at 8pm',
      );

      expect(result['error'], contains('T1'));
      expect(find.text('Table T1 reserved'), findsNothing);
      await finish(tester);
    });

    testWidgets('refuses a booking whose summary does not name the table', (tester) async {
      await open(tester);
      await show(tester, [
        _card('m1', [('T1', 'Cliff Terrace', 2, true), ('T4', 'Harbor Window', 4, true)]),
      ]);
      await settle(tester, brought, 'The card never brought its tool.');

      final result = await call(
        tester,
        'T4',
        'Harbor Window',
        summary: 'Cliff Terrace, for 2, tonight at 8pm',
      );

      expect(result['error'], contains('T4'));
      expect(find.text('Table T4 reserved'), findsNothing);
      await finish(tester);
    });
  });
}
