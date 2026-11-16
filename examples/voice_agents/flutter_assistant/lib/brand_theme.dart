// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai_a2ui/stream_chat_flutter_ai_a2ui.dart';

/// Brand design-system presets applied to both OpenUI and A2UI generated surfaces as well as
/// custom portable components.
enum BrandProfile {
  stream(
    title: 'Stream Default',
    subtitle: 'Electric blue accent · 16px rounded cards',
    accent: Color(0xFF005FFF),
    cornerRadius: 16,
  ),
  ember(
    title: 'Ember Hospitality',
    subtitle: 'Warm terracotta accent · 10px compact cards',
    accent: Color(0xFFD95D24),
    cornerRadius: 10,
  ),
  nordic(
    title: 'Nordic Mobility',
    subtitle: 'Crisp teal accent · 6px architectural edges',
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

  A2UIThemeData get a2uiTheme => A2UIThemeData(accent: accent, cornerRadius: cornerRadius);

  static final ValueNotifier<BrandProfile> active = ValueNotifier(BrandProfile.stream);
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
