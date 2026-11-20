// ignore: deprecated_member_use
import 'package:flutter/material.dart';

/// Eases its size to its child's as the child grows or shrinks, or follows it at once when
/// [animates] is false or the phone asks for reduced motion.
///
/// [AnimatedSize] can't follow at once by itself: given no duration, it changes its layout in
/// the middle of laying out, which Flutter refuses, so following at once takes the shortest
/// duration there is.
class SmoothSize extends StatelessWidget {
  const SmoothSize({
    super.key,
    required this.duration,
    this.curve = Curves.linear,
    this.animates = true,
    required this.child,
  });

  final Duration duration;
  final Curve curve;

  /// Whether a change of size eases in; otherwise it shows on the next frame.
  final bool animates;

  final Widget child;

  static const _atOnce = Duration(microseconds: 1);

  @override
  Widget build(BuildContext context) => AnimatedSize(
    duration: animates && !MediaQuery.disableAnimationsOf(context) ? duration : _atOnce,
    curve: curve,
    alignment: Alignment.topCenter,
    child: child,
  );
}
