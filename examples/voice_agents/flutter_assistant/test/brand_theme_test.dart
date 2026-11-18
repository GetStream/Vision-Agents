// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_assistant/brand_theme.dart';
import 'package:flutter_assistant/openui_renderer.dart';
import 'package:flutter_assistant/trip_components.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:shared_preferences_platform_interface/in_memory_shared_preferences_async.dart';
import 'package:shared_preferences_platform_interface/shared_preferences_async_platform_interface.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'support/ui_message.dart';

void main() {
  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    await TripComponents.load();
  });

  tearDown(() => BrandProfile.active.value = BrandProfile.stream);

  group('BrandProfile', () {
    test('load restores the brand chosen at an earlier launch and keeps the next one', () async {
      final store = InMemorySharedPreferencesAsync.withData({'brand': 'nordic'});
      SharedPreferencesAsyncPlatform.instance = store;

      await BrandProfile.load();
      expect(BrandProfile.active.value, BrandProfile.nordic);

      BrandProfile.active.value = BrandProfile.ember;
      final preferences = SharedPreferencesAsync();
      final deadline = DateTime.now().add(const Duration(seconds: 5));
      while (await preferences.getString('brand') != 'ember') {
        if (DateTime.now().isAfter(deadline)) fail('The chosen brand was not kept.');
      }
    });

    testWidgets('a card already shown takes on a newly chosen brand', (tester) async {
      final host = AIUIHost(
        components: TripComponents.all,
        renderers: [OpenUIRenderer()],
        send: (_) async {},
      );
      addTearDown(host.dispose);
      final reply = UIMessage('m1', '''
root = RestaurantCard("Kaneo", "Macedonian Grill", 4.9, "\$\$\$")
''');
      host.receive([reply]);
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(extensions: [StreamTheme.light()]),
          home: Scaffold(
            body: BrandThemeScope(
              child: AIUIView(host: host, document: reply.uiDocument, messageId: reply.id),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();
      Color? badgeColor() => tester.widget<Text>(find.textContaining('★ 4.9')).style?.color;
      expect(badgeColor(), BrandProfile.stream.accent);

      BrandProfile.active.value = BrandProfile.ember;
      await tester.pumpAndSettle();

      expect(badgeColor(), BrandProfile.ember.accent);
    });
  });
}
