import 'dart:convert';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:stream_agents/stream_agents.dart';

/// Photos from the composer, as the router takes them.
///
/// The composer hands over files on the phone; a question takes an image by URL, and a phone
/// has nowhere public to put one. So each is shrunk and sent inline, as a data URI: a photo
/// straight off the camera is several megabytes, which is more than a model needs to read it.
abstract final class Photos {
  /// The longest side a photo is sent at, in pixels.
  static const _longestSide = 1280;

  /// Each of [files], ready to ask with.
  ///
  /// Throws a [PhotoFailure] when one does not read as an image, rather than asking without
  /// it: the person sees it in their question, and the model would say it sees none.
  static Future<List<AgentImage>> sources(List<XFile> files) async {
    final images = <AgentImage>[];
    final unread = <XFile>[];
    for (final file in files) {
      final jpeg = await _jpeg(file);
      if (jpeg == null) {
        unread.add(file);
      } else {
        images.add(AgentImage('data:image/jpeg;base64,${base64Encode(jpeg)}'));
      }
    }
    if (unread.isNotEmpty) throw PhotoFailure(unread);
    return images;
  }

  /// The photo as a JPEG no longer than [_longestSide] on either side, turned the way the
  /// camera held it, or null when it does not read as an image.
  static Future<List<int>?> _jpeg(XFile file) async {
    final ui.ImageDescriptor descriptor;
    try {
      descriptor = await ui.ImageDescriptor.encoded(
        await ui.ImmutableBuffer.fromFilePath(file.path),
      );
    } on Exception {
      return null;
    }
    final longest = max(descriptor.width, descriptor.height);
    final shortest = min(descriptor.width, descriptor.height);
    descriptor.dispose();
    // The compressor scales until both sides reach the minimum, so asking for the short
    // side's share of the longest side brings the long one to it, whichever way the photo
    // is turned.
    final side = longest <= _longestSide ? shortest : (shortest * _longestSide / longest).round();
    try {
      return await FlutterImageCompress.compressWithFile(
        file.path,
        minWidth: side,
        minHeight: side,
        quality: 70,
        keepExif: false,
      );
      // CompressError and UnsupportedError are Errors, not Exceptions.
    } on Object {
      return null;
    }
  }
}

/// Photos that could not be read, so the question was not asked.
class PhotoFailure implements Exception {
  const PhotoFailure(this.files);

  /// The files that did not read as an image.
  final List<XFile> files;

  @override
  String toString() =>
      'Could not read ${files.map((file) => file.name).join(', ')} as a photo. '
      'Send again without it.';
}
