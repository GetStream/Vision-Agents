import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as image;
import 'package:vision_agents_core/vision_agents_core.dart';

/// A picture the camera took, as the JPEG the agent is sent, made off the UI isolate.
///
/// Stream Video hands a picture over at the camera's full size, as a JPEG on Android and a PNG
/// on iOS, too large to send eight of over the socket.
Future<Uint8List> cameraJpeg(Uint8List picture) => compute(jpegOf, picture);

/// [picture] as a JPEG at most [CameraFrames.longestSide] on its longest side, at
/// [CameraFrames.quality]. Throws a [FormatException] for something that is not a picture.
@visibleForTesting
Uint8List jpegOf(Uint8List picture) {
  final decoded = image.decodeImage(picture);
  if (decoded == null) {
    throw const FormatException('not a picture the camera could have taken');
  }
  final scale = min(1, CameraFrames.longestSide / max(decoded.width, decoded.height));
  final scaled = scale == 1
      ? decoded
      : image.copyResize(
          decoded,
          width: (decoded.width * scale).round(),
          height: (decoded.height * scale).round(),
          interpolation: image.Interpolation.average,
        );
  return image.encodeJpg(scaled, quality: CameraFrames.quality);
}
