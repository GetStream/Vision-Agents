import 'dart:convert';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:stream_core_flutter/core.dart';

import 'assistant.dart';
import 'brand_theme.dart';

/// How the agent answers: in Markdown, or with interactive UI in A2UI or OpenUI.
///
/// The backend keeps it, as what the agent is told, so the app has nothing to switch: its
/// conversations show any UI it can draw, and a reply without UI as its text.
enum ReplyFormat {
  markdown('Markdown'),
  a2ui('A2UI'),
  openui('OpenUI');

  const ReplyFormat(this.title);

  final String title;

  static Uri _url(Uri? backend) => (backend ?? Assistant.backendUrl).resolve('/reply-format');

  /// The format the agent answers in, asked of [backend], or null for one the app does not
  /// offer, such as OpenUI, which another app chose.
  static Future<ReplyFormat?> current({Uri? backend}) async =>
      _read(await http.get(_url(backend)).timeout(_answering));

  /// Has the agent answer in this format, from its next conversation on.
  Future<void> choose({Uri? backend}) async => _read(
    await http
        .put(
          _url(backend),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'format': name}),
        )
        .timeout(_answering),
  );

  static const _answering = Duration(seconds: 15);

  static ReplyFormat? _read(http.Response response) {
    if (response.statusCode != 200) {
      final said = response.body.trim();
      throw Exception(said.isEmpty ? "The backend didn't answer." : said);
    }
    final format = (jsonDecode(response.body) as Map<String, Object?>)['format'];
    return ReplyFormat.values.asNameMap()[format];
  }
}

/// The app's settings: how the agent answers.
class SettingsSheet extends StatefulWidget {
  const SettingsSheet({super.key});

  static Future<void> show(BuildContext context) => showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (context) => const SettingsSheet(),
  );

  @override
  State<SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<SettingsSheet> {
  ReplyFormat? _format;
  bool _loaded = false;
  bool _choosing = false;
  String? _failure;

  @override
  void initState() {
    super.initState();
    ReplyFormat.current().then(
      (format) => mounted
          ? setState(() {
              _format = format;
              _loaded = true;
            })
          : null,
      // A format can still be chosen without knowing the current one.
      onError: (Object error) => mounted
          ? setState(() {
              _failure = '$error';
              _loaded = true;
            })
          : null,
    );
  }

  /// One at a time, so a failed choice goes back to the format that was really chosen.
  Future<void> _choose(ReplyFormat format) async {
    final was = _format;
    setState(() {
      _format = format;
      _failure = null;
      _choosing = true;
    });
    try {
      await format.choose();
    } on Object catch (error) {
      if (mounted) {
        setState(() {
          _format = was;
          _failure = '$error';
        });
      }
    } finally {
      if (mounted) setState(() => _choosing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: Text('Replies', style: text.headingSm.copyWith(color: colors.textPrimary)),
          ),
          for (final format in ReplyFormat.values)
            StreamListTile(
              title: Text(format.title),
              trailing: format == _format ? Icon(Icons.check, color: colors.accentPrimary) : null,
              onTap: _loaded && !_choosing ? () => _choose(format) : null,
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              _failure ??
                  'New conversations answer in this format. OpenUI and A2UI replies show '
                      'interactive maps, restaurant cards, gauges, and forms.',
              style: text.captionDefault.copyWith(
                color: _failure == null ? colors.textSecondary : colors.accentError,
              ),
            ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: Text(
              'Brand Design Tokens',
              style: text.headingSm.copyWith(color: colors.textPrimary),
            ),
          ),
          ValueListenableBuilder<BrandProfile>(
            valueListenable: BrandProfile.active,
            builder: (context, active, _) => Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                for (final profile in BrandProfile.values)
                  StreamListTile(
                    title: Text(profile.title),
                    subtitle: Text(profile.subtitle),
                    trailing: profile == active ? Icon(Icons.check, color: profile.accent) : null,
                    onTap: () => BrandProfile.active.value = profile,
                  ),
              ],
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
