import 'dart:async';

/// A value that changes, readable now and watchable: Dart's answer to Kotlin's `StateFlow`.
///
/// [value] is what it is now. [stream] hands every listener the current value first and then
/// each change, so a widget built from it never shows a blank frame and never misses the
/// state it subscribed in. A listener that pauses, which `await for` does while its body
/// awaits, is not queued every change it missed: when it resumes it gets the latest, which is
/// all a view wants. Pure Dart, so the Flutter packages adapt it to `ValueListenable` and
/// `StreamBuilder` rather than the core depending on Flutter.
abstract interface class LiveValue<T> {
  T get value;

  /// The current value, then every change, conflated for a listener that falls behind.
  Stream<T> get stream;
}

/// The writable side of a [LiveValue], kept by whoever owns the state.
final class LiveValueController<T> implements LiveValue<T> {
  LiveValueController(this._value);

  T _value;
  final Set<_Listener<T>> _listeners = {};
  bool _closed = false;

  @override
  T get value => _value;

  /// Replaces the value, telling every listener unless it is the same one.
  set value(T next) {
    if (_closed || next == _value) {
      return;
    }
    _value = next;
    for (final listener in [..._listeners]) {
      listener.changed();
    }
  }

  @override
  Stream<T> get stream => Stream.multi((controller) {
    if (_closed) {
      controller
        ..add(_value)
        ..close();
      return;
    }
    final listener = _Listener<T>(controller, () => _value);
    _listeners.add(listener);
    controller
      ..onResume = listener.resumed
      ..onCancel = () => _listeners.remove(listener);
    controller.add(_value);
  });

  /// Ends every stream. The value stays readable.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    for (final listener in [..._listeners]) {
      listener.controller.close();
    }
    _listeners.clear();
  }
}

final class _Listener<T> {
  _Listener(this.controller, this.read);

  final MultiStreamController<T> controller;
  final T Function() read;
  bool _missed = false;

  void changed() {
    if (controller.isPaused) {
      _missed = true;
    } else {
      controller.add(read());
    }
  }

  void resumed() {
    if (_missed) {
      _missed = false;
      controller.add(read());
    }
  }
}
