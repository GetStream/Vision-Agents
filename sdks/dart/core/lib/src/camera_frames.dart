import 'dart:typed_data';

/// What this device's camera showed lately, kept so the agent can look when it asks.
///
/// The router asks with `get_video_frames` when its vision skill needs to see, naming the
/// moment it took the question and how many frames it wants. While [AgentSession.cameraFrames]
/// is set, the session answers from here. `VoiceSession` in `vision_agents_rtc` samples the
/// call's camera twice a second while it is on and keeps the newest eight: about four seconds.
/// Nothing here touches a camera, so an app with a camera of its own can [record] into one too.
final class CameraFrames {
  /// [encode] turns a picture as it was recorded into the JPEG the agent is sent, and runs
  /// only for the frames an answer sends. Without it every picture is taken to be one already,
  /// and should be small: up to [keep] go in one socket frame, and the router closes a socket
  /// whose frame passes 5 MiB.
  CameraFrames({Future<Uint8List> Function(Uint8List picture)? encode})
    : encode = encode ?? _already;

  /// The tool the router asks for frames with. Answered by the session, never offered to the
  /// model as one of the app's tools.
  static const toolName = 'get_video_frames';

  /// How often a frame is kept.
  static const every = Duration(milliseconds: 500);

  /// How many are kept.
  static const keep = 8;

  /// How old the newest may be at the moment asked about, as the router's own workers allow.
  static const freshFor = Duration(seconds: 5);

  /// How far this device's clock may run ahead of the router's: a frame taken just after the
  /// question by this clock may be just before it by the router's.
  static const skew = Duration(seconds: 2);

  /// The longest side a frame is sent at, in pixels.
  static const longestSide = 1024;

  /// The JPEG quality a frame is sent at, out of 100.
  static const quality = 70;

  final Future<Uint8List> Function(Uint8List picture) encode;

  final List<CameraFrame> _kept = [];
  int _counter = 0;

  /// Keeps one picture the camera took at [capturedAt], dropping the oldest past [keep].
  void record(Uint8List picture, DateTime capturedAt) {
    _kept.add(CameraFrame._(++_counter, capturedAt, picture));
    if (_kept.length > keep) {
      _kept.removeRange(0, _kept.length - keep);
    }
  }

  /// Forgets every frame, for a camera turned off: what it showed before is not what it shows.
  void clear() => _kept.clear();

  /// The newest [limit] frames taken at or before [at], oldest first, as the router's own
  /// workers pick them: never a newer frame in place of one the question was asked about.
  ///
  /// Empty when the camera has shown nothing recent enough. [limit] is held to 1 through
  /// [keep].
  List<CameraFrame> select(DateTime at, int limit) {
    final before = [
      for (final frame in _kept)
        if (!frame.capturedAt.isAfter(at.add(skew))) frame,
    ];
    if (before.isEmpty || at.difference(before.last.capturedAt) > freshFor) {
      return const [];
    }
    return before.sublist(before.length - limit.clamp(1, keep).clamp(1, before.length));
  }

  static Future<Uint8List> _already(Uint8List picture) async => picture;
}

/// One picture the camera took.
final class CameraFrame {
  CameraFrame._(this.id, this.capturedAt, this.picture);

  /// Counts up from 1 for the life of the store, so the agent can tell frames apart.
  final int id;
  final DateTime capturedAt;

  /// The picture as it was recorded, before [CameraFrames.encode].
  final Uint8List picture;
}
