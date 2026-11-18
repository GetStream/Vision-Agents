// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:stream_chat_flutter_ai_a2ui/stream_chat_flutter_ai_a2ui.dart';

/// Brand design-system presets applied to both OpenUI and A2UI generated surfaces as well as
/// custom portable components.
enum BrandProfile {
  stream(
    title: 'Stream Default',
    subtitle: 'Electric blue accent · 16pt rounded cards',
    accent: Color(0xFF005FFF),
    cornerRadius: 16,
  ),
  ember(
    title: 'Ember Hospitality',
    subtitle: 'Warm terracotta accent · 10pt compact cards',
    accent: Color(0xFFD95D24),
    cornerRadius: 10,
  ),
  nordic(
    title: 'Nordic Mobility',
    subtitle: 'Crisp teal accent · 6pt architectural edges',
    accent: Color(0xFF0F766E),
    cornerRadius: 6,
  );

  const BrandProfile({
    required this.title,
    required this.subtitle,
    required this.accent,
    required this.cornerRadius,
  });

  final String title;
  final String subtitle;
  final Color accent;
  final double cornerRadius;

  A2UIThemeData get a2uiTheme => A2UIThemeData(
    accent: accent,
    cornerRadius: cornerRadius,
    buttonCornerRadius: cornerRadius * 0.75,
    controlCornerRadius: cornerRadius / 2,
  );

  /// The brand chosen in Settings, kept between launches once [load] has run.
  static final ValueNotifier<BrandProfile> active = ValueNotifier(BrandProfile.stream);

  /// Where the chosen brand is kept, by its name.
  static const _storageKey = 'brand';

  /// Reads the brand chosen at an earlier launch into [active], and keeps each one chosen from
  /// now on.
  ///
  /// A brand that can't be read leaves the default in place: the app opens all the same.
  static Future<void> load() async {
    final preferences = SharedPreferencesAsync();
    try {
      final stored = await preferences.getString(_storageKey);
      active.value = values.asNameMap()[stored] ?? active.value;
    } on PlatformException catch (error) {
      debugPrint('Reading the chosen brand failed: $error');
    } on TypeError catch (error) {
      // Saved by an older build as something other than a name.
      debugPrint('The saved brand was not a name: $error');
    }
    active.addListener(() async {
      try {
        await preferences.setString(_storageKey, active.value.name);
      } on PlatformException catch (error) {
        debugPrint('Saving the chosen brand failed: $error');
      }
    });
  }
}

/// Inherited brand tokens read by OpenUI components and custom portable widgets.
class BrandThemeData extends InheritedWidget {
  const BrandThemeData({super.key, required this.profile, required super.child});

  final BrandProfile profile;

  static BrandProfile of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<BrandThemeData>()?.profile ??
      BrandProfile.active.value;

  @override
  bool updateShouldNotify(BrandThemeData oldWidget) => oldWidget.profile != profile;
}

/// Applies the active [BrandProfile] to OpenUI and A2UI subtrees.
class BrandThemeScope extends StatelessWidget {
  const BrandThemeScope({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<BrandProfile>(
      valueListenable: BrandProfile.active,
      builder: (context, profile, _) => BrandThemeData(
        profile: profile,
        child: A2UITheme(data: profile.a2uiTheme, child: child),
      ),
    );
  }
}
