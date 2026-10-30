import 'dart:async';
import 'dart:typed_data';

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_rtc/src/camera_shots.dart';

void main() {
  group('CameraShots', () {
    final asked = DateTime.utc(2026, 10, 27, 12);

    /// A camera whose track is [track], taking each picture with [take].
    CameraShot? Function() camera(Future<Uint8List?> Function() take, {Object track = 'back'}) =>
        () => (track: track, take: take);

    /// Leaves each picture as it was taken.
    Future<Uint8List> asTaken(Uint8List picture) async => picture;

    List<List<int>> jpegsOf(List<CameraPicture> pictures) => [
      for (final picture in pictures) picture.jpeg,
    ];

    test('takes as many pictures as asked for, one after another', () {
      fakeAsync((async) {
        var taken = 0;
        final shots = CameraShots(camera(() async => Uint8List.fromList([taken += 1])), asTaken);
        List<CameraPicture>? looked;

        shots.look(asked, 3).then((pictures) => looked = pictures);
        async.elapse(const Duration(seconds: 2));

        expect(jpegsOf(looked!), [
          [1],
          [2],
          [3],
        ]);
      });
    });

    test('waits half a second after one picture before taking the next', () {
      fakeAsync((async) {
        var asking = 0;
        final shots = CameraShots(camera(() async => Uint8List.fromList([asking += 1])), asTaken);

        shots.look(asked, 2);
        async.elapse(CameraShots.between - const Duration(milliseconds: 1));
        expect(asking, 1);

        async.elapse(const Duration(milliseconds: 1));
        expect(asking, 2);
      });
    });

    test('makes each picture what the encoder gives', () async {
      final shots = CameraShots(
        camera(() async => Uint8List.fromList([1])),
        (picture) async => Uint8List.fromList([9, ...picture]),
      );

      final [picture] = await shots.look(asked, 1);

      expect(picture.jpeg, [9, 1]);
    });

    test('has nothing while there is no camera sending video', () async {
      final shots = CameraShots(() => null, asTaken);

      expect(await shots.look(asked, 3), isEmpty);
    });

    test('leaves out a picture the platform could not take', () {
      fakeAsync((async) {
        var asking = 0;
        final shots = CameraShots(
          camera(
            () async => (asking += 1) == 1 ? throw StateError('no frame') : Uint8List.fromList([2]),
          ),
          asTaken,
        );
        List<CameraPicture>? looked;

        shots.look(asked, 2).then((pictures) => looked = pictures);
        async.elapse(const Duration(seconds: 1));

        expect(jpegsOf(looked!), [
          [2],
        ]);
      });
    });

    test('asks a track that still owes a picture for nothing more', () {
      fakeAsync((async) {
        final owed = Completer<Uint8List?>();
        var asking = 0;
        final shots = CameraShots(
          camera(() {
            asking += 1;
            return owed.future;
          }),
          asTaken,
        );

        shots
          ..look(asked, 1)
          ..look(asked, 1);
        async.flushMicrotasks();

        expect(asking, 1);
      });
    });

    test('asks a track that replaced one still owing a picture', () {
      fakeAsync((async) {
        var track = 'back';
        final asking = <String>[];
        final shots = CameraShots(
          () => (
            track: track,
            take: () {
              asking.add(track);
              return Completer<Uint8List?>().future;
            },
          ),
          asTaken,
        );

        shots.look(asked, 1);
        track = 'back-2';
        shots.look(asked, 1);
        async.flushMicrotasks();

        expect(asking, ['back', 'back-2']);
      });
    });

    test('has nothing once stopped, even what it already took', () {
      fakeAsync((async) {
        final shots = CameraShots(camera(() async => Uint8List.fromList([1])), asTaken);
        List<CameraPicture>? looked;

        shots.look(asked, 3).then((pictures) => looked = pictures);
        async.elapse(const Duration(milliseconds: 600));
        shots.stop();
        async.elapse(const Duration(seconds: 1));

        expect(looked, isEmpty);
      });
    });

    test('takes no more pictures once stopped', () {
      fakeAsync((async) {
        var asking = 0;
        final shots = CameraShots(camera(() async => Uint8List.fromList([asking += 1])), asTaken);

        shots.look(asked, 3);
        async.flushMicrotasks();
        shots.stop();
        async.elapse(const Duration(seconds: 2));

        expect(asking, 1);
      });
    });

    test('shares a picture still owed from before it stopped with the next look', () {
      fakeAsync((async) {
        final owed = Completer<Uint8List?>();
        var asking = 0;
        final shots = CameraShots(
          camera(() {
            asking += 1;
            return owed.future;
          }),
          asTaken,
        );
        List<CameraPicture>? looked;

        shots.look(asked, 1);
        async.flushMicrotasks();
        shots.stop();
        shots.look(asked, 1).then((pictures) => looked = pictures);
        owed.complete(Uint8List.fromList([1]));
        async.flushMicrotasks();

        expect(asking, 1);
        expect(jpegsOf(looked!), [
          [1],
        ]);
      });
    });
  });
}
