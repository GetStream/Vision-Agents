import 'package:flutter/material.dart';

/// A screen's whole content when there is nothing else to show: an icon, a title and a line
/// saying why.
class Notice extends StatelessWidget {
  const Notice({super.key, required this.icon, required this.title, required this.message});

  final IconData icon;
  final String title;
  final String message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        spacing: 8,
        children: [
          Icon(icon, size: 56, color: theme.colorScheme.onSurfaceVariant),
          Text(title, style: theme.textTheme.titleLarge, textAlign: TextAlign.center),
          Text(
            message,
            style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
