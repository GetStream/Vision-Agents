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
  static const pace = 15.0;

  /// How long a caption stays up once nothing new is being said.
  static const linger = Duration(milliseconds: 2500);

  /// How long the agent has to be quiet, with its reply written, for the reply to be over.
  static const finished = Duration(milliseconds: 1500);

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
          (lastAudible == null || now - lastAudible > finished);
      final paced =
          (_spoken.inMicroseconds / Duration.microsecondsPerSecond * pace)
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
        (agentIsAudible || (lastChange != null && now - lastChange < linger));
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

  /// The end of [text] that fits in [maxLines] laid out by [painter]: the words before it are
  /// cut, behind an ellipsis, so the newest words always show.
  static String fitEnd(
    String text, {
    required TextPainter painter,
    required double width,
    required int maxLines,
  }) {
    final words = text.split(' ');
    String from(int word) =>
        word == 0 ? text : '…${words.skip(word).join(' ')}';
    bool fits(int word) {
      painter
        ..text = TextSpan(text: from(word), style: painter.text?.style)
        ..maxLines = maxLines
        ..layout(maxWidth: width);
      return !painter.didExceedMaxLines;
    }

    // The fewest words cut that fit, found by halving: cutting more never stops it fitting.
    var low = 0;
    var high = words.length - 1;
    while (low < high) {
      final middle = (low + high) ~/ 2;
      if (fits(middle)) {
        high = middle;
      } else {
        low = middle + 1;
      }
    }
    return from(low);
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

  static const _maxLines = 2;

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
            // The newest words are what is being said, so a caption too long for two lines
            // loses its start rather than its end.
            child: LayoutBuilder(
              builder: (context, constraints) {
                final textStyle = style?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.w500,
                  fontStyle: caption?.isPerson ?? false
                      ? FontStyle.italic
                      : FontStyle.normal,
                );
                final painter = TextPainter(
                  text: TextSpan(style: textStyle),
                  textAlign: TextAlign.center,
                  textDirection: Directionality.of(context),
                  textScaler: MediaQuery.textScalerOf(context),
                );
                final text = Captions.fitEnd(
                  caption?.text ?? '',
                  painter: painter,
                  width: constraints.maxWidth,
                  maxLines: _maxLines,
                );
                painter.dispose();
                return Text(
                  text,
                  maxLines: _maxLines,
                  overflow: TextOverflow.ellipsis,
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
