// core, chat and rtc are kept apart inside the package: each area is self-contained under
// lib/src/<area> and test/<area>, core stays pure Dart, and the only way in from another area is
// its barrel. These tests keep that true: they read the directives of every file under lib/ and
// test/ and fail on one that reaches where it should not.
//
// Run from the package root, as flutter test does.
import 'dart:io';

import 'package:test/test.dart';

const _areas = ['core', 'chat', 'rtc'];

/// What each area may import from another: core nothing, chat and rtc core's barrel.
const _reaches = {
  'core': <String>[],
  'chat': ['core'],
  'rtc': ['core'],
};

/// The third-party packages each is the one area to use. A package in no list is shared.
const _owned = {
  'chat': ['stream_chat'],
  'rtc': ['stream_video_flutter', 'image'],
};

/// What core keeps out besides the other areas' packages, so it stays pure Dart and runs in a
/// browser: Flutter, and the libraries a browser does not have.
const _notInCore = ['package:flutter/', 'package:flutter_test/', 'dart:io', 'dart:ui'];

final _directive = RegExp(r'''^\s*(?:import|export|part)\s+['"]([^'"]+)['"]''', multiLine: true);

/// The import, export and part URIs of every Dart file under [directory], by path.
Map<String, List<String>> _directives(String directory) => {
  for (final file in Directory(directory).listSync(recursive: true).whereType<File>())
    if (file.path.endsWith('.dart'))
      file.path: [
        for (final match in _directive.allMatches(file.readAsStringSync())) match.group(1)!,
      ],
};

/// The area a package:stream_agents URI is in, or null for the public library.
String? _areaOf(String uri) => RegExp(r'^package:stream_agents/src/(\w+)/').firstMatch(uri)?[1];

/// The package a package: URI is from.
String? _packageOf(String uri) => RegExp(r'^package:([^/]+)/').firstMatch(uri)?[1];

void main() {
  test('every file under lib/src is in an area', () {
    final stray = [
      for (final path in _directives('lib/src').keys)
        if (!_areas.any((area) => path.startsWith('lib/src/$area/'))) path,
    ];
    expect(stray, isEmpty);
  });

  test('the public library exports the three area barrels and nothing else', () {
    expect(_directives('lib')['lib/stream_agents.dart'], [
      'src/chat/chat.dart',
      'src/core/core.dart',
      'src/rtc/rtc.dart',
    ]);
  });

  for (final area in _areas) {
    group(area, () {
      final files = _directives('lib/src/$area');

      test('reaches another area only through its barrel, and only the areas it may', () {
        final wrong = [
          for (final MapEntry(key: path, value: uris) in files.entries)
            for (final uri in uris)
              if (_areaOf(uri) case final other? when other != area)
                if (!_reaches[area]!.contains(other) ||
                    uri != 'package:stream_agents/src/$other/$other.dart')
                  '$path: $uri',
        ];
        expect(wrong, isEmpty);
      });

      test('imports its own files relatively, and never the public library', () {
        final wrong = [
          for (final MapEntry(key: path, value: uris) in files.entries)
            for (final uri in uris)
              if (_areaOf(uri) == area ||
                  uri.startsWith('package:stream_agents/stream_agents.dart') ||
                  // A relative path out of the area is another area without its barrel.
                  _leavesArea(path, uri, area))
                '$path: $uri',
        ];
        expect(wrong, isEmpty);
      });

      test('uses no third-party package another area owns', () {
        final wrong = [
          for (final MapEntry(key: path, value: uris) in files.entries)
            for (final uri in uris)
              if (_ownerOf(uri) case final owner? when owner != area) '$path: $uri',
        ];
        expect(wrong, isEmpty);
      });

      if (area == 'core') {
        test('stays pure Dart', () {
          final wrong = [
            for (final MapEntry(key: path, value: uris) in files.entries)
              for (final uri in uris)
                if (_notInCore.any(uri.startsWith)) '$path: $uri',
          ];
          expect(wrong, isEmpty);
        });
      }
    });

    group('test/$area', () {
      final files = _directives('test/$area');

      test('imports its own area or the public library, and no other area', () {
        final wrong = [
          for (final MapEntry(key: path, value: uris) in files.entries)
            for (final uri in uris)
              if ((_packageOf(uri) == 'stream_agents' &&
                      uri != 'package:stream_agents/stream_agents.dart' &&
                      _areaOf(uri) != area) ||
                  _leavesArea(path, uri, area))
                '$path: $uri',
        ];
        expect(wrong, isEmpty);
      });

      test('uses no third-party package another area owns', () {
        final wrong = [
          for (final MapEntry(key: path, value: uris) in files.entries)
            for (final uri in uris)
              if (_ownerOf(uri) case final owner? when owner != area) '$path: $uri',
        ];
        expect(wrong, isEmpty);
      });
    });
  }
}

/// The area that owns the package [uri] is from, or null when none does.
String? _ownerOf(String uri) {
  final package = _packageOf(uri);
  for (final MapEntry(key: area, value: packages) in _owned.entries) {
    if (packages.contains(package)) return area;
  }
  return null;
}

/// Whether a relative [uri] in the file at [path] climbs out of [area]'s directory.
bool _leavesArea(String path, String uri, String area) {
  if (uri.contains(':')) return false;
  final root = path.startsWith('lib/') ? 'lib/src/$area' : 'test/$area';
  final resolved = File(path).parent.uri.resolve(uri).path;
  return !resolved.startsWith('$root/');
}
