import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as image;
import 'package:vision_agents_rtc/src/camera_jpeg.dart';

/// A picture of [width] by [height] with something in it, as a camera would hand one over.
image.Image _picture(int width, int height) {
  final picture = image.Image(width: width, height: height);
  for (final pixel in picture) {
    pixel
      ..r = pixel.x % 256
      ..g = pixel.y % 256
      ..b = 128;
  }
  return picture;
}

void main() {
  group('cameraJpeg', () {
    test("makes iOS's full-size PNG a JPEG 1024 wide, the same shape", () async {
      final jpeg = await cameraJpeg(image.encodePng(_picture(1920, 1080)));

      final decoded = image.JpegDecoder().decode(jpeg)!;
      expect((decoded.width, decoded.height), (1024, 576));
    });

    test("makes Android's full-size portrait JPEG one 1024 tall", () async {
      final jpeg = await cameraJpeg(image.encodeJpg(_picture(720, 1280), quality: 100));

      final decoded = image.JpegDecoder().decode(jpeg)!;
      expect((decoded.width, decoded.height), (576, 1024));
    });

    test('keeps a picture already small enough at its size', () async {
      final picture = image.encodeJpg(_picture(640, 480), quality: 100);

      final jpeg = await cameraJpeg(picture);

      final decoded = image.JpegDecoder().decode(jpeg)!;
      expect((decoded.width, decoded.height), (640, 480));
    });

    test('makes a picture already small enough a smaller JPEG', () async {
      final picture = image.encodeJpg(_picture(640, 480), quality: 100);

      final jpeg = await cameraJpeg(picture);

      expect(jpeg.length, lessThan(picture.length));
    });

    test('refuses something that is not a picture', () async {
      await expectLater(cameraJpeg(Uint8List.fromList([1, 2, 3])), throwsFormatException);
    });
  });
}
