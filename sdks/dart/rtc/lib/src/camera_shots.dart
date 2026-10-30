import 'dart:async';

import 'package:flutter/foundation.dart';

/// One picture the camera can be asked for: [track] names the camera track it comes from, and
/// [take] asks for it.
typedef CameraShot = ({Object? track, Future<Uint8List?> Function() take});

/// A picture for the agent: a JPEG, and when the camera was asked for it.
typedef CameraPicture = ({Uint8List jpeg, DateTime capturedAt});

/// Takes the pictures the agent asks for with this device's camera, at the moment it asks.
///
/// Nothing is taken until the agent asks, and nothing is kept. Stream Video takes each
/// picture as a screenshot of the call's own video, which waits for the camera's next frame
/// and cannot be called off: a picture asked of a track that still owes one shares that one
/// rather than asking again.
final class CameraShots {
  /// [camera] gives the picture to take now, or null when there is no camera sending video;
  /// [encode] makes each picture into the JPEG the agent is sent.
  CameraShots(this._camera, this._encode);

  /// How far apart the pictures are when the agent asks for more than one, so they show what
  /// moved.
  static const between = Duration(milliseconds: 500);

  final CameraShot? Function() _camera;
  final Future<Uint8List> Function(Uint8List picture) _encode;

  /// The pictures each track still owes.
  final Map<Object?, Future<Uint8List?>> _owed = {};

  /// Counts [stop]s, so a look under way can tell it was stopped.
  int _generation = 0;

  /// Takes [limit] pictures, [between] apart, starting now.
  ///
  /// [at], the moment the question was taken, is not looked back to: nothing from before the
  /// agent asked is kept. A picture the camera could not take or that is not a picture is left
  /// out. Stops at a camera that is no longer sending video, and comes back with nothing at all
  /// once [stop] is called.
  Future<List<CameraPicture>> look(DateTime at, int limit) async {
    final generation = _generation;
    final pictures = <CameraPicture>[];
    for (var taken = 0; taken < limit; taken += 1) {
      if (taken > 0) {
        await Future<void>.delayed(between);
      }
      final shot = _camera();
      if (generation != _generation || shot == null) {
        break;
      }
      // Stamped when asked rather than when it arrives: the picture is the next frame the
      // camera gives.
      final capturedAt = DateTime.now();
      try {
        final picture = await _take(shot);
        if (picture != null && generation == _generation) {
          pictures.add((jpeg: await _encode(picture), capturedAt: capturedAt));
        }
      } catch (_) {
        // A picture not taken, for whatever reason the platform had, is one picture fewer.
      }
    }
    // What a camera turned off showed is not for the agent.
    return generation == _generation ? pictures : const [];
  }

  /// Stops every look under way. A picture a track still owes stays owed, so a camera turned
  /// back on shares it rather than asking for a second.
  void stop() => _generation += 1;

  Future<Uint8List?> _take(CameraShot shot) {
    final track = shot.track;
    // A block, not an arrow: an action returning the removed future would wait on itself.
    return _owed[track] ??= Future.sync(shot.take).whenComplete(() {
      _owed.remove(track);
    });
  }
}
