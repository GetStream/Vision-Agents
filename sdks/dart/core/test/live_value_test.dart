import 'dart:async';

import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  group('LiveValueController', () {
    test('a listener gets the value it subscribed in, then each change', () async {
      final value = LiveValueController(1);
      final seen = <int>[];
      final subscription = value.stream.listen(seen.add);

      await pumpEventQueue();
      value.value = 2;
      value.value = 3;
      await pumpEventQueue();

      expect(seen, [1, 2, 3]);
      await subscription.cancel();
    });

    test('setting the same value tells nobody', () async {
      final value = LiveValueController('a');
      final seen = <String>[];
      final subscription = value.stream.listen(seen.add);

      await pumpEventQueue();
      value.value = 'a';
      await pumpEventQueue();

      expect(seen, ['a']);
      await subscription.cancel();
    });

    test('a listener that paused gets the latest on resuming, not everything it missed', () async {
      final value = LiveValueController(0);
      final seen = <int>[];
      final subscription = value.stream.listen(seen.add);
      await pumpEventQueue();

      subscription.pause();
      for (var i = 1; i <= 50; i++) {
        value.value = i;
      }
      subscription.resume();
      await pumpEventQueue();

      expect(seen, [0, 50]);
      await subscription.cancel();
    });

    test('two listeners each see every change', () async {
      final value = LiveValueController(0);
      final first = <int>[];
      final second = <int>[];
      final a = value.stream.listen(first.add);
      final b = value.stream.listen(second.add);

      await pumpEventQueue();
      value.value = 1;
      await pumpEventQueue();

      expect(first, [0, 1]);
      expect(second, [0, 1]);
      await a.cancel();
      await b.cancel();
    });

    test('closing ends every stream and keeps the value readable', () async {
      final value = LiveValueController(7);
      final done = Completer<void>();
      value.stream.listen(null, onDone: done.complete);
      await pumpEventQueue();

      value.close();
      value.value = 8;

      await done.future.timeout(const Duration(seconds: 1));
      expect(value.value, 7);
      expect(await value.stream.toList(), [7]);
    });
  });
}
