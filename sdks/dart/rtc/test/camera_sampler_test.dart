import 'dart:async';
import 'dart:typed_data';

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';
import 'package:vision_agents_rtc/src/camera_sampler.dart';

void main() {
  group('CameraSampler', () {
    late CameraFrames frames;

    setUp(() => frames = CameraFrames());

    List<List<int>> kept() => [
      for (final frame in frames.select(DateTime.now(), CameraFrames.keep)) frame.picture,
    ];

    /// A camera whose track is [track], taking each picture with [take].
    CameraShot? Function() camera(Future<Uint8List?> Function() take, {Object track = 'back'}) =>
        () => (track: track, take: take);

    test('keeps a picture every half second, the newest eight', () {
      fakeAsync((async) {
        var taken = 0;
        final sampler = CameraSampler(camera(() async => Uint8List.fromList([++taken])), frames);

        async.elapse(const Duration(seconds: 5));

        expect(kept(), [
          [3],
          [4],
          [5],
          [6],
          [7],
          [8],
          [9],
          [10],
        ]);
        sampler.stop();
      });
    });

    test('skips a picture due while the last is still being taken', () {
      fakeAsync((async) {
        var taken = 0;
        final sampler = CameraSampler(
          camera(
            () => Future.delayed(
              const Duration(milliseconds: 1200),
              () => Uint8List.fromList([++taken]),
            ),
          ),
          frames,
        );

        async.elapse(const Duration(seconds: 3));

        expect(kept(), [
          [1],
        ]);
        sampler.stop();
      });
    });

    test('keeps nothing taken after it stopped', () {
      fakeAsync((async) {
        final picture = Completer<Uint8List?>();
        final sampler = CameraSampler(camera(() => picture.future), frames);
        async.elapse(const Duration(milliseconds: 500));

        sampler.stop();
        picture.complete(Uint8List.fromList([1]));
        async.flushMicrotasks();

        expect(kept(), isEmpty);
      });
    });

    test('gives up on a picture that never comes, and takes them from the track after it', () {
      fakeAsync((async) {
        var asked = 0;
        final sampler = CameraSampler(() {
          final replaced = asked > 0;
          return (
            track: replaced ? 'back-2' : 'back',
            take: () => ++asked == 1
                ? Completer<Uint8List?>().future
                : Future.value(Uint8List.fromList([asked])),
          );
        }, frames);

        async.elapse(const Duration(seconds: 4));

        expect(kept(), isNotEmpty);
        expect(kept().first, [2]);
        sampler.stop();
      });
    });

    test('asks a stalled track for nothing more until it gives the picture it owes', () {
      fakeAsync((async) {
        final owed = Completer<Uint8List?>();
        var asked = 0;
        final sampler = CameraSampler(
          camera(() => ++asked == 1 ? owed.future : Future.value(Uint8List.fromList([asked]))),
          frames,
        );

        // Ten seconds of a stalled camera: one picture asked for, not one every few seconds.
        async.elapse(const Duration(seconds: 10));
        expect(asked, 1);
        expect(kept(), isEmpty);

        owed.complete(Uint8List.fromList([1]));
        async.elapse(const Duration(seconds: 1));

        expect(asked, 3);
        expect(kept(), [
          [2],
          [3],
        ]);
        sampler.stop();
      });
    });

    test('goes on after a picture the platform could not take', () {
      fakeAsync((async) {
        var asked = 0;
        final sampler = CameraSampler(
          camera(
            () async => ++asked == 1 ? throw StateError('no frame') : Uint8List.fromList([asked]),
          ),
          frames,
        );

        async.elapse(const Duration(seconds: 1));

        expect(kept(), [
          [2],
        ]);
        sampler.stop();
      });
    });

    test('keeps nothing while there is no camera sending video', () {
      fakeAsync((async) {
        final sampler = CameraSampler(() => null, frames);

        async.elapse(const Duration(seconds: 2));

        expect(kept(), isEmpty);
        sampler.stop();
      });
    });
  });
}
