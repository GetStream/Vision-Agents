// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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

    test(
      'load opens with the default brand when the saved one cannot be read, and keeps the next',
      () async {
        final store = _UnreadableStore();
        SharedPreferencesAsyncPlatform.instance = store;

        await BrandProfile.load();
        expect(BrandProfile.active.value, BrandProfile.stream);

        BrandProfile.active.value = BrandProfile.nordic;
        final deadline = DateTime.now().add(const Duration(seconds: 5));
        while (store.saved['brand'] != 'nordic') {
          if (DateTime.now().isAfter(deadline)) fail('The chosen brand was not kept.');
          await Future<void>.delayed(const Duration(milliseconds: 5));
        }
      },
    );

    test('a brand that cannot be saved does not fail the app', () async {
      final store = _UnwritableStore();
      SharedPreferencesAsyncPlatform.instance = store;
      await BrandProfile.load();

      BrandProfile.active.value = BrandProfile.ember;

      // The save was tried, and its failure would surface as an error nobody handled.
      final deadline = DateTime.now().add(const Duration(seconds: 5));
      while (store.tried['brand'] != 'ember') {
        if (DateTime.now().isAfter(deadline)) fail('The chosen brand was never saved.');
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
      await pumpEventQueue();
    });

    test('load opens with the default brand when the saved one is of another type', () async {
      SharedPreferencesAsyncPlatform.instance = _MistypedStore();

      await BrandProfile.load();

      expect(BrandProfile.active.value, BrandProfile.stream);
      // The next brand chosen is still kept.
      BrandProfile.active.value = BrandProfile.nordic;
      final preferences = SharedPreferencesAsync();
      final deadline = DateTime.now().add(const Duration(seconds: 5));
      while (await preferences.getString('brand') != 'nordic') {
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

/// A store whose saved brand can't be read, as when the phone's storage fails, and which keeps
/// what is written to it.
base class _UnreadableStore extends InMemorySharedPreferencesAsync {
  _UnreadableStore() : super.empty();

  final saved = <String, String>{};

  @override
  Future<String?> getString(String key, SharedPreferencesOptions options) =>
      throw PlatformException(code: 'unreadable');

  @override
  Future<bool> setString(String key, String value, SharedPreferencesOptions options) async {
    saved[key] = value;
    return true;
  }
}

/// A store that can't be written to, as when the phone's storage is full.
base class _UnwritableStore extends InMemorySharedPreferencesAsync {
  _UnwritableStore() : super.empty();

  /// What was tried to be saved.
  final tried = <String, String>{};

  @override
  Future<bool> setString(String key, String value, SharedPreferencesOptions options) {
    tried[key] = value;
    throw PlatformException(code: 'unwritable');
  }
}

/// A store whose `brand` was saved as a number, as by an older build.
base class _MistypedStore extends InMemorySharedPreferencesAsync {
  _MistypedStore() : super.withData({'brand': 3});
}
