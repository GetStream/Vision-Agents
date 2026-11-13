// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

/// A screen's whole content when there is nothing else to show: an icon, a title and a line
/// saying why.
class Notice extends StatelessWidget {
  const Notice({super.key, required this.icon, required this.title, required this.message});

  final IconData icon;
  final String title;
  final String message;

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final spacing = context.streamSpacing;
    return Padding(
      padding: EdgeInsets.all(spacing.xl),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        spacing: spacing.xs,
        children: [
          Icon(icon, size: 56, color: colors.textSecondary),
          Text(
            title,
            style: text.headingSm.copyWith(color: colors.textPrimary),
            textAlign: TextAlign.center,
          ),
          Text(
            message,
            style: text.bodyDefault.copyWith(color: colors.textSecondary),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
