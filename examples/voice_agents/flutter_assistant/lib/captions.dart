import 'package:flutter/material.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// What a caption says, and whether the person said it.
@immutable
class Caption {
  const Caption(this.text, {required this.isPerson});

  final String text;
  final bool isPerson;

  @override
  bool operator ==(Object other) =>
      other is Caption && other.text == text && other.isPerson == isPerson;

  @override
  int get hashCode => Object.hash(text, isPerson);
}

/// Closed captions for a spoken conversation: what is being said, a sentence at a time, gone a
/// moment after the talking stops.
///
/// The agent's reply is written before it is spoken, so its words are revealed at the pace of
/// its voice, counting the time its audio is heard, rather than as they arrive. What the
/// person said appears once it has been transcribed.
class Captions {
  /// About how fast the agent speaks, in characters a second.
  static const _pace = 15.0;

  /// How long a caption stays up once nothing new is being said.
  static const _linger = Duration(milliseconds: 2500);

  /// How long the agent has to be quiet, with its reply written, for the reply to be over.
  static const _finished = Duration(milliseconds: 1500);

  Caption? _caption;
  bool _isVisible = false;
  String? _turnId;
  Duration _spoken = Duration.zero;
  Duration? _lastAudible;
  Duration? _lastChange;

  Caption? get caption => _caption;
  bool get isVisible => _isVisible;

  /// Moves the captions on to [now], [elapsed] after the last update.
  void update({
    required Turn? turn,
    required bool agentIsAudible,
    required bool agentIsAnswering,
    required Duration now,
    required Duration elapsed,
  }) {
    Caption? next;
    if (turn != null && turn.speaker.isAgent) {
      if (turn.id != _turnId) {
        _turnId = turn.id;
        _spoken = Duration.zero;
      }
      if (agentIsAudible) {
        _spoken += elapsed;
        _lastAudible = now;
      }
      // Once the agent has stopped for good, whatever the pacing has not reached yet is
      // shown, so a reply never ends half captioned.
      final lastAudible = _lastAudible;
      final isOver =
          _spoken > Duration.zero &&
          !agentIsAnswering &&
          (lastAudible == null || now - lastAudible > _finished);
      final paced =
          (_spoken.inMicroseconds / Duration.microsecondsPerSecond * _pace)
              .floor();
      final count = isOver
          ? turn.text.length
          : paced.clamp(0, turn.text.length);
      final said = sentence(turn.text, count);
      next = said.isEmpty ? null : Caption(said, isPerson: false);
    } else if (turn != null) {
      final said = sentence(turn.text, turn.text.length);
      next = said.isEmpty ? null : Caption(said, isPerson: true);
    }
    if (next != null && next != _caption) {
      _caption = next;
      _lastChange = now;
    }
    final lastChange = _lastChange;
    _isVisible =
        _caption != null &&
        (agentIsAudible || (lastChange != null && now - lastChange < _linger));
  }

  /// The sentence being said once the first [count] characters of [text] have been: whole
  /// words only, starting after the last sentence that has ended.
  static String sentence(String text, int count) {
    var said = text.substring(0, count);
    if (count < text.length) {
      // Hold back a word until all of it has been said.
      final space = said.lastIndexOf(RegExp(r'\s'));
      said = space < 0 ? '' : said.substring(0, space);
    }
    var start = 0;
    for (var i = 0; i < said.length - 1; i++) {
      final ends = '.!?'.contains(said[i]) || said[i] == '\n';
      if (ends && said[i + 1].trim().isEmpty) start = i + 1;
    }
    return said.substring(start).trim();
  }
}

/// A caption, white on a dark band at the bottom of the screen like a film's. What the person
/// said is in italics.
class CaptionView extends StatelessWidget {
  const CaptionView({
    super.key,
    required this.caption,
    required this.isVisible,
  });

  final Caption? caption;
  final bool isVisible;

  /// The end of [text] that fits in two lines of [maxWidth], led by an ellipsis when its start
  /// is left out: what is being said now, rather than how the sentence began.
  static String tail(
    String text, {
    required TextStyle? style,
    required double maxWidth,
    TextScaler textScaler = TextScaler.noScaling,
  }) {
    bool fits(String candidate) {
      final painter = TextPainter(
        text: TextSpan(text: candidate, style: style),
        maxLines: 2,
        textDirection: TextDirection.ltr,
        textScaler: textScaler,
      )..layout(maxWidth: maxWidth);
      final fits = !painter.didExceedMaxLines;
      painter.dispose();
      return fits;
    }

    if (fits(text)) return text;
    final words = text.split(' ');
    for (var start = 1; start < words.length; start++) {
      final candidate = '…${words.sublist(start).join(' ')}';
      if (fits(candidate)) return candidate;
    }
    return '…${words.last}';
  }

  @override
  Widget build(BuildContext context) {
    final style = Theme.of(context).textTheme.bodyLarge;
    // Two lines' room whether or not a caption is up, so the glow never moves.
    return SizedBox(
      height: 72,
      width: double.infinity,
      child: Align(
        alignment: Alignment.bottomCenter,
        child: AnimatedOpacity(
          opacity: isVisible ? 1 : 0,
          duration: const Duration(milliseconds: 350),
          curve: Curves.easeOut,
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.7),
              borderRadius: BorderRadius.circular(10),
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final textStyle = style?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.w500,
                  fontStyle: caption?.isPerson ?? false
                      ? FontStyle.italic
                      : FontStyle.normal,
                );
                return Text(
                  tail(
                    caption?.text ?? '',
                    style: textStyle,
                    maxWidth: constraints.maxWidth,
                    textScaler: MediaQuery.textScalerOf(context),
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.clip,
                  textAlign: TextAlign.center,
                  style: textStyle,
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}
