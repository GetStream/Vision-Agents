// core, chat and rtc are kept apart inside the package: each area is self-contained under
// lib/src/<area> and test/<area>, core and chat stay pure Dart, and the only way in from another
// area is its barrel. These tests keep that true: they read the directives of every file under
// lib/ and test/ and fail on one that reaches where it should not.
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

/// What core may import besides its own files: the libraries every platform has, a browser
/// included, and the packages that pick the platform themselves. A list of what is allowed
/// rather than of what is not, so a dependency added for rtc, or a Flutter plugin, cannot slip
/// into core.
const _portable = [
  'dart:async',
  'dart:collection',
  'dart:convert',
  'dart:math',
  'dart:typed_data',
  'package:http/',
  'package:meta/',
  'package:web_socket/',
];

/// What each pure-Dart area may import besides its own files.
const _pureDart = {
  'core': _portable,
  'chat': [..._portable, 'package:stream_chat/', 'package:stream_agents/src/core/core.dart'],
};

/// What a pure-Dart area's tests keep out, so they would run without Flutter: Flutter itself,
/// and the public library, which brings rtc and Stream Video with it.
const _notInPureTests = [
  'package:flutter/',
  'package:flutter_test/',
  'package:stream_agents/stream_agents.dart',
];

/// One directive, across lines, up to its semicolon: every quoted URI in it counts, so a
/// conditional import's alternatives are read too.
final _directive = RegExp(r'^\s*(?:import|export|part)\b[^;]*;', multiLine: true);
final _quoted = RegExp(r'''['"]([^'"]+)['"]''');
final _areaUri = RegExp(r'^package:stream_agents/src/(\w+)/');
final _packageUri = RegExp(r'^package:([^/]+)/');

/// The package's root, which flutter test runs from.
final _root = Directory.current.absolute.uri;

/// The import, export and part URIs of every Dart file under [directory], by its path from the
/// package root with forward slashes.
Map<String, List<String>> _directives(String directory) => {
  for (final file in Directory.fromUri(_root.resolve(directory)).listSync(recursive: true))
    if (file is File && file.path.endsWith('.dart'))
      file.absolute.uri.path.substring(_root.path.length): [
        for (final directive in _directive.allMatches(file.readAsStringSync()))
          for (final uri in _quoted.allMatches(directive[0]!)) uri[1]!,
      ],
};

/// The area a package:stream_agents URI is in, or null for the public library.
String? _areaOf(String uri) => _areaUri.firstMatch(uri)?[1];

/// The package a package: URI is from.
String? _packageOf(String uri) => _packageUri.firstMatch(uri)?[1];

void main() {
  test('every file under lib/src is in an area', () {
    final stray = [
      for (final path in _directives('lib/src/').keys)
        if (!_areas.any((area) => path.startsWith('lib/src/$area/'))) path,
    ];
    expect(stray, isEmpty);
  });

  test('every test is in an area, but this one', () {
    final stray = [
      for (final path in _directives('test/').keys)
        if (path != 'test/area_boundaries_test.dart' &&
            !_areas.any((area) => path.startsWith('test/$area/')))
          path,
    ];
    expect(stray, isEmpty);
  });

  test('the public library exports the three area barrels and nothing else', () {
    expect(_directives('lib/')['lib/stream_agents.dart'], [
      'src/chat/chat.dart',
      'src/core/core.dart',
      'src/rtc/rtc.dart',
    ]);
  });

  for (final area in _areas) {
    group(area, () {
      final files = _directives('lib/src/$area/');

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

      if (_pureDart[area] case final allowed?) {
        test('stays pure Dart: imports only what runs on every platform', () {
          final wrong = [
            for (final MapEntry(key: path, value: uris) in files.entries)
              for (final uri in uris)
                if (uri.contains(':') && !allowed.any(uri.startsWith)) '$path: $uri',
          ];
          expect(wrong, isEmpty);
        });
      }
    });

    group('test/$area', () {
      final files = _directives('test/$area/');

      test('imports its own area, the barrels it reaches or the public library, and nothing else '
          'of the package', () {
        final wrong = [
          for (final MapEntry(key: path, value: uris) in files.entries)
            for (final uri in uris)
              if ((_packageOf(uri) == 'stream_agents' &&
                      uri != 'package:stream_agents/stream_agents.dart' &&
                      _areaOf(uri) != area &&
                      !_reaches[area]!.any(
                        (other) => uri == 'package:stream_agents/src/$other/$other.dart',
                      )) ||
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

      if (_pureDart.containsKey(area)) {
        test('runs without Flutter', () {
          final wrong = [
            for (final MapEntry(key: path, value: uris) in files.entries)
              for (final uri in uris)
                if (_notInPureTests.any(uri.startsWith)) '$path: $uri',
          ];
          expect(wrong, isEmpty);
        });
      }
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
  final resolved = Uri.parse(path).resolve(uri).path;
  return !resolved.startsWith('$root/');
}
