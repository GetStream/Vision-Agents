import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// One picture the camera can be asked for: [track] names the camera track it comes from, and
/// [take] asks for it.
typedef CameraShot = ({Object? track, Future<Uint8List?> Function() take});

/// Takes a picture with this device's camera every [CameraFrames.every] while it is on, for
/// the agent to look at, until [stop].
///
/// Stream Video has no hook on the frames a call sends from Flutter, so [VoiceSession] asks
/// for each as a screenshot of the call's own video: the camera's next frame, at its full
/// size. One at a time: a picture due while the last is still being taken is skipped rather
/// than queued, and one not there within [patience] is given up on. Giving up does not stop
/// the platform waiting for that frame, so a track that has not given the last picture asked
/// of it is asked for nothing more until it does: a camera that stalls, as it does while the
/// app is in the background, holds one request rather than one more every few seconds. A
/// track that was replaced never gives it, and its replacement is asked as usual.
final class CameraSampler {
  CameraSampler(this._camera, this._frames, {this.patience = const Duration(seconds: 2)}) {
    _timer = Timer.periodic(CameraFrames.every, (_) => _sample());
  }

  /// How long a picture may take before it is given up on.
  final Duration patience;

  /// The picture to take now, or null when there is no camera sending video to take it with.
  final CameraShot? Function() _camera;
  final CameraFrames _frames;
  late final Timer _timer;
  bool _taking = false;
  bool _stopped = false;

  /// The track a picture was asked of that has not come yet, or null.
  Object? _waitingOn;

  void stop() {
    _stopped = true;
    _timer.cancel();
  }

  Future<void> _sample() async {
    if (_taking) {
      return;
    }
    final shot = _camera();
    if (shot == null || (_waitingOn != null && _waitingOn == shot.track)) {
      return;
    }
    _taking = true;
    _waitingOn = shot.track;
    // Stamped when asked rather than when it arrives. The picture is the next frame the camera
    // gives, at once while it is running; one that waited on a paused camera only looks older
    // than it is, and is too old to be shown.
    final at = DateTime.now();
    final picture = Future.sync(shot.take);
    unawaited(
      picture.then<void>((_) {}, onError: (Object _) {}).whenComplete(() {
        if (_waitingOn == shot.track) {
          _waitingOn = null;
        }
      }),
    );
    try {
      final taken = await picture.timeout(patience);
      // A camera turned off while this was taken showed nothing the agent should see.
      if (taken != null && !_stopped) {
        _frames.record(taken, at);
      }
    } catch (_) {
      // A picture not taken, for whatever reason the platform had, is a frame not kept; the
      // call's own video is untouched, and the next one is tried as usual.
    } finally {
      _taking = false;
    }
  }
}
