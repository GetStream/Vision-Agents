import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// Takes a picture with this device's camera every [CameraFrames.every] while it is on, for
/// the agent to look at, until [stop].
///
/// Stream Video has no hook on the frames a call sends from Flutter, so [VoiceSession] asks
/// for each as a screenshot of the call's own video: the camera's next frame, at its full
/// size. One at a time: a picture due while the last is still being taken is skipped rather
/// than queued, and one not there within [patience] is given up on, since a camera track that
/// was stopped or replaced never gives the frame it was asked for.
final class CameraSampler {
  CameraSampler(this._take, this._frames, {this.patience = const Duration(seconds: 2)}) {
    _timer = Timer.periodic(CameraFrames.every, (_) => _sample());
  }

  /// How long a picture may take before it is given up on.
  final Duration patience;

  /// Takes one picture, or null when there is no camera to take it with.
  final Future<Uint8List?> Function() _take;
  final CameraFrames _frames;
  late final Timer _timer;
  bool _taking = false;
  bool _stopped = false;

  void stop() {
    _stopped = true;
    _timer.cancel();
  }

  Future<void> _sample() async {
    if (_taking) {
      return;
    }
    _taking = true;
    // Stamped when asked rather than when it arrives. The picture is the next frame the camera
    // gives, at once while it is running; one that waited on a paused camera only looks older
    // than it is, and is too old to be shown.
    final at = DateTime.now();
    try {
      final picture = await _take().timeout(patience);
      // A camera turned off while this was taken showed nothing the agent should see.
      if (picture != null && !_stopped) {
        _frames.record(picture, at);
      }
    } catch (_) {
      // A picture not taken, for whatever reason the platform had, is a frame not kept; the
      // call's own video is untouched, and the next one is tried as usual.
    } finally {
      _taking = false;
    }
  }
}
