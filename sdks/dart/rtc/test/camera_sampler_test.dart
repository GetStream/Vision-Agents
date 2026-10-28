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

    test('keeps a picture every half second, the newest eight', () {
      fakeAsync((async) {
        var taken = 0;
        final sampler = CameraSampler(() async => Uint8List.fromList([++taken]), frames);

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
          () => Future.delayed(
            const Duration(milliseconds: 1200),
            () => Uint8List.fromList([++taken]),
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
        final sampler = CameraSampler(() => picture.future, frames);
        async.elapse(const Duration(milliseconds: 500));

        sampler.stop();
        picture.complete(Uint8List.fromList([1]));
        async.flushMicrotasks();

        expect(kept(), isEmpty);
      });
    });

    test('gives up on a picture that never comes, and goes on taking them', () {
      fakeAsync((async) {
        var asked = 0;
        final sampler = CameraSampler(
          () => ++asked == 1
              ? Completer<Uint8List?>().future
              : Future.value(Uint8List.fromList([asked])),
          frames,
        );

        async.elapse(const Duration(seconds: 3));

        expect(kept(), isNotEmpty);
        expect(kept().first, [2]);
        sampler.stop();
      });
    });

    test('goes on after a picture the platform could not take', () {
      fakeAsync((async) {
        var asked = 0;
        final sampler = CameraSampler(
          () async => ++asked == 1 ? throw StateError('no frame') : Uint8List.fromList([asked]),
          frames,
        );

        async.elapse(const Duration(seconds: 1));

        expect(kept(), [
          [2],
        ]);
        sampler.stop();
      });
    });

    test('keeps nothing while there is no camera to take a picture with', () {
      fakeAsync((async) {
        final sampler = CameraSampler(() async => null, frames);

        async.elapse(const Duration(seconds: 2));

        expect(kept(), isEmpty);
        sampler.stop();
      });
    });
  });
}
