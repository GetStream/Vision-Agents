import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as image;

/// Makes [picture] into the JPEG the agent is sent, off the UI isolate where the platform has one.
///
/// Stream Video hands a picture over at the camera's full size, as a JPEG on Android and a PNG
/// on iOS, too large to send eight of over the socket.
Future<Uint8List> cameraJpeg(Uint8List picture) => compute(_jpegOf, picture);

/// The longest side a picture is sent at, in pixels.
const _longestSide = 1024;

/// The JPEG quality a picture is sent at, out of 100.
const _quality = 70;

/// [picture] as a JPEG at most [_longestSide] on its longest side, at [_quality]. Throws a
/// [FormatException] for something that is not a picture.
Uint8List _jpegOf(Uint8List picture) {
  // Only the two formats Stream Video gives, so an app does not compile in every decoder
  // `package:image` has. Checked first: decoding a PNG as a JPEG throws.
  final jpeg = image.JpegDecoder();
  final decoded = jpeg.isValidFile(picture)
      ? jpeg.decode(picture)
      : image.PngDecoder().decode(picture);
  if (decoded == null) {
    throw const FormatException('not a picture the camera could have taken');
  }
  final scale = min(1, _longestSide / max(decoded.width, decoded.height));
  final scaled = scale == 1
      ? decoded
      : image.copyResize(
          decoded,
          width: (decoded.width * scale).round(),
          height: (decoded.height * scale).round(),
          interpolation: image.Interpolation.average,
        );
  return image.encodeJpg(scaled, quality: _quality);
}
