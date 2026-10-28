import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  group('CameraFrames', () {
    final asked = DateTime.utc(2026, 10, 27, 12);
    late CameraFrames frames;

    setUp(() => frames = CameraFrames());

    void recordAt(List<int> seconds) {
      for (final second in seconds) {
        frames.record(Uint8List.fromList([second + 10]), asked.add(Duration(seconds: second)));
      }
    }

    List<int> idsOf(List<CameraFrame> selected) => [for (final frame in selected) frame.id];

    test('picks the newest it was asked for, taken at or before the moment, oldest first', () {
      recordAt([-3, -2, -1, 0]);

      expect(idsOf(frames.select(asked, 2)), [3, 4]);
      // Two seconds' allowance for the clocks: asked three seconds ago, a second ago is the newest.
      expect(idsOf(frames.select(asked.subtract(const Duration(seconds: 3)), 2)), [2, 3]);
    });

    test('takes a frame up to two seconds after the moment, for a clock running ahead', () {
      recordAt([-1, 2, 3]);

      expect(idsOf(frames.select(asked, 8)), [1, 2]);
    });

    test('has nothing when the newest is more than five seconds older than the moment', () {
      recordAt([-6]);
      expect(frames.select(asked, 1), isEmpty);

      recordAt([-5]);
      expect(idsOf(frames.select(asked, 1)), [2]);
    });

    test('has nothing before the camera showed anything', () {
      expect(frames.select(asked, 1), isEmpty);
    });

    test('gives at least one frame and at most what it keeps', () {
      recordAt([-4, -3, -2, -1, 0]);

      expect(idsOf(frames.select(asked, 0)), [5]);
      expect(idsOf(frames.select(asked, -3)), [5]);
      expect(idsOf(frames.select(asked, 20)), [1, 2, 3, 4, 5]);
    });

    test('keeps the newest eight, about four seconds at two a second', () {
      for (var i = 0; i < 10; i++) {
        frames.record(Uint8List(0), asked.add(Duration(milliseconds: 500 * i - 5000)));
      }

      expect(idsOf(frames.select(asked, 8)), [3, 4, 5, 6, 7, 8, 9, 10]);
    });

    test('forgets what the camera showed when cleared, and counts on', () {
      recordAt([-1]);
      frames.clear();
      expect(frames.select(asked, 1), isEmpty);

      recordAt([0]);
      final [frame] = frames.select(asked, 1);
      expect(frame.id, 2);
      expect(frame.picture, [10]);
      expect(frame.capturedAt, asked);
    });

    test('takes each picture for a JPEG already unless it is given an encoder', () async {
      expect(await CameraFrames().encode(Uint8List.fromList([1, 2])), [1, 2]);
    });
  });
}
