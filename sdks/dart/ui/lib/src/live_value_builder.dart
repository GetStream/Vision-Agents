import 'dart:async';

import 'package:flutter/widgets.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// Rebuilds whenever a [LiveValue] changes, starting from its current value.
///
/// A `StreamBuilder` handed the stream alone draws one frame with no data first; this never
/// does, because the value is read before the stream has said anything.
class LiveValueBuilder<T> extends StatelessWidget {
  const LiveValueBuilder({super.key, required this.value, required this.builder});

  final LiveValue<T> value;
  final Widget Function(BuildContext context, T value) builder;

  @override
  Widget build(BuildContext context) => StreamBuilder<T>(
    stream: value.stream,
    initialData: value.value,
    builder: (context, snapshot) => builder(context, snapshot.data as T),
  );
}

/// A [LiveValue] as a [ValueListenable], for a host that holds state in `ValueNotifier`s,
/// `ListenableBuilder` or anything built on `Listenable`.
///
/// It listens from the moment it is made until [dispose].
class LiveValueNotifier<T> extends ValueNotifier<T> {
  LiveValueNotifier(LiveValue<T> source) : super(source.value) {
    _subscription = source.stream.listen((next) => value = next);
  }

  late final StreamSubscription<T> _subscription;

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }
}
