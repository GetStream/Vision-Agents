// Regenerates lib/src/generated/api.dart from the acceleration OpenAPI spec.
//
// The spec at acceleration/api/openapi.yaml is the source of truth. The output is committed,
// so depending on the package needs no code generation and no build_runner.
//
// Only the operations a device is allowed to call are generated. The filter is the point: an
// operation the spec does not mark x-client-accessible is server-side only, and a client SDK
// with a method for one would only be offering its callers a 403. --check verifies the filter
// still agrees with the spec and that the committed output is what the spec generates.
//
// The session events socket is audited but not generated: OpenAPI stops at the upgrade, so it
// is hand-written in lib/src/socket.dart.
//
//     dart run tool/generate.dart
//     dart run tool/generate.dart --check
import 'dart:io';

import 'package:yaml/yaml.dart';

// What a device may reach: opening, finding, reading back, rewinding, forking and ending a
// conversation, a guest to hold one as, and looking something up. Everything else about an
// agent is its backend's to ask for and hand down.
const operations = [
  'closeSession',
  'createGuestUser',
  'createResponse',
  'createSession',
  'forkSession',
  'getSession',
  'listResponseItems',
  'listResponses',
  'listSessions',
  'rewindSession',
  'search',
  'searchSessions',
];

// Client accessible, and hand-written because they are sockets.
const sockets = ['watchSession'];

final root = File.fromUri(Platform.script).parent.parent;
final spec = File('${root.path}/../../../acceleration/api/openapi.yaml');
final target = File('${root.path}/lib/src/generated/api.dart');

void main(List<String> args) {
  if (!spec.existsSync()) {
    stderr.writeln('no spec at ${spec.path}');
    exit(1);
  }
  final document = loadYaml(spec.readAsStringSync()) as YamlMap;

  final complaints = audit(document);
  if (complaints.isNotEmpty) {
    complaints.forEach(stderr.writeln);
    exit(1);
  }

  final generated = format(Generator(document).emit());
  if (args.contains('--check')) {
    if (!target.existsSync() || target.readAsStringSync() != generated) {
      stderr.writeln('${target.path} is stale: run dart run tool/generate.dart');
      exit(1);
    }
    print(
      '${operations.length} client operations and ${sockets.length} socket, '
      'none server-side only, output up to date',
    );
    return;
  }

  target.parent.createSync(recursive: true);
  target.writeAsStringSync(generated);
  print('regenerated lib/src/generated/api.dart');
}

/// The named operations the spec does not open to a client, or does not have.
List<String> audit(YamlMap document) {
  final found = <String, YamlMap>{};
  for (final item in (document['paths'] as YamlMap).values) {
    for (final operation in (item as YamlMap).values) {
      if (operation is YamlMap && operation['operationId'] is String) {
        found[operation['operationId'] as String] = operation;
      }
    }
  }
  return [
    for (final name in [...operations, ...sockets])
      if (found[name] == null)
        '$name is not in the spec'
      else if (found[name]!['x-client-accessible'] != true)
        '$name is server-side only and cannot be in a client SDK',
  ];
}

/// Runs the output through dart format, so a regenerated file and a committed one compare
/// byte for byte.
String format(String source) {
  final scratch = Directory.systemTemp.createTempSync('vision_agents_generate');
  try {
    final file = File('${scratch.path}/api.dart')..writeAsStringSync(source);
    final result = Process.runSync('dart', [
      'format',
      '--language-version=3.10',
      '--page-width=100',
      file.path,
    ]);
    if (result.exitCode != 0) {
      stderr.writeln(result.stderr);
      exit(1);
    }
    return file.readAsStringSync();
  } finally {
    scratch.deleteSync(recursive: true);
  }
}

class Operation {
  Operation(this.id, this.method, this.path, this.definition);

  final String id;
  final String method;
  final String path;
  final YamlMap definition;
}

/// How one schema is held in Dart, and how a value of it crosses the wire.
class Kind {
  Kind(this.type, this.decode, this.encode);

  final String type;

  /// Turns a JSON expression into a Dart value, given where it was read for the error.
  final String Function(String value, String at) decode;

  /// Turns a Dart value back into JSON.
  final String Function(String value) encode;
}

class Generator {
  Generator(this.document)
    : schemas = (document['components'] as YamlMap)['schemas'] as YamlMap,
      parameters = (document['components'] as YamlMap)['parameters'] as YamlMap;

  final YamlMap document;
  final YamlMap schemas;
  final YamlMap parameters;
  final Set<String> classes = {};

  String emit() {
    final found = <Operation>[];
    for (final MapEntry(key: path, value: item) in (document['paths'] as YamlMap).entries) {
      for (final MapEntry(key: method, value: operation) in (item as YamlMap).entries) {
        if (operation is YamlMap && operations.contains(operation['operationId'])) {
          found.add(
            Operation(
              operation['operationId'] as String,
              method as String,
              path as String,
              operation,
            ),
          );
        }
      }
    }
    found.sort((a, b) => a.id.compareTo(b.id));

    final methods = [for (final operation in found) method(operation)];
    final written = <String, String>{};
    while (classes.difference(written.keys.toSet()).isNotEmpty) {
      final next = (classes.difference(written.keys.toSet()).toList()..sort()).first;
      written[next] = model(next);
    }
    final models = [for (final name in written.keys.toList()..sort()) written[name]!];

    final info = document['info'] as YamlMap;
    return '''
// Generated by tool/generate.dart from acceleration/api/openapi.yaml (${info['title']} ${info['version']}).
// Do not edit: change the spec and regenerate.
//
// Internal to the package. Every type crossing the public API is hand-written in models.dart,
// so the spec can churn without breaking a caller.

import '../errors.dart';
import '../wire.dart';

/// The client operations, one method each.
final class Operations {
  Operations(this._wire);

  final Wire _wire;

${methods.join('\n')}
}

${models.join('\n')}

${helpers.trim()}
''';
  }

  String method(Operation operation) {
    final arguments = <String>[];
    final query = <String>[];
    var path = operation.path;

    for (final raw in (operation.definition['parameters'] as YamlList? ?? YamlList())) {
      final parameter = resolveParameter(raw as YamlMap);
      final name = parameter['name'] as String;
      final kind = kindOf(parameter['schema'] as YamlMap);
      final dart = identifier(name);
      if (parameter['in'] == 'path') {
        arguments.add('required ${kind.type} $dart');
        path = path.replaceAll('{$name}', '\${Uri.encodeComponent($dart)}');
      } else {
        arguments.add('${kind.type}? $dart');
        query.add("'$name': ${queryValue(kind, dart)}");
      }
    }

    final body = operation.definition['requestBody'] as YamlMap?;
    if (body != null) {
      final schema = jsonSchema(body)!;
      final kind = kindOf(schema);
      arguments.add(body['required'] == true ? 'required ${kind.type} body' : '${kind.type}? body');
    }

    final success = successSchema(operation.definition);
    final returns = success == null ? 'void' : kindOf(success).type;
    final call = [
      "'${operation.method.toUpperCase()}'",
      "'$path'",
      "operation: '${operation.id}'",
      if (query.isNotEmpty) 'query: {${query.join(', ')}}',
      if (body != null) body['required'] == true ? 'body: body.toJson()' : 'body: body?.toJson()',
    ].join(', ');

    final signature = arguments.isEmpty ? '' : '{${arguments.join(', ')}}';
    final summary = operation.definition['summary'];
    final doc = summary == null ? '' : '  /// $summary.\n';
    if (success == null) {
      return '$doc  Future<void> ${operation.id}($signature) async {\n'
          '    await _wire.send($call);\n'
          '  }\n';
    }
    return '$doc  Future<$returns> ${operation.id}($signature) async {\n'
        '    final json = await _wire.send($call);\n'
        '    return ${kindOf(success).decode('json', operation.id)};\n'
        '  }\n';
  }

  String model(String name) {
    final schema = schemas[name] as YamlMap;
    final properties = schema['properties'] as YamlMap? ?? YamlMap();
    final required = {...(schema['required'] as YamlList? ?? YamlList()).cast<String>()};

    final fields = <String>[];
    final parameters = <String>[];
    final reads = <String>[];
    final writes = <String>[];
    for (final MapEntry(key: wire, value: property) in properties.entries) {
      final key = wire as String;
      final kind = kindOf(property as YamlMap);
      final dart = identifier(key);
      final at = '$name.$key';
      if (required.contains(key)) {
        fields.add('  final ${kind.type} $dart;');
        parameters.add('required this.$dart');
        reads.add("$dart: ${kind.decode("json['$key']", at)}");
        writes.add("'$key': ${kind.encode(dart)}");
      } else {
        fields.add('  final ${kind.type}? $dart;');
        parameters.add('this.$dart');
        reads.add(
          "$dart: switch (json['$key']) { null => null, final Object value => ${kind.decode('value', at)} }",
        );
        final encoded = kind.encode('value');
        writes.add(
          encoded == 'value' ? "'$key': ?$dart" : "if ($dart case final value?) '$key': $encoded",
        );
      }
    }

    final description = schema['description'];
    final doc = description is String ? '/// ${firstSentence(description)}\n' : '';
    final constructor = parameters.isEmpty
        ? '  const $name();'
        : '  const $name({${parameters.join(', ')}});';
    return '''
$doc final class $name {
$constructor

  factory $name.fromJson(Object? value) {
    final json = _object(value, '$name');
    return $name(${reads.map((read) => '$read,').join('\n')});
  }

${fields.join('\n')}

  Map<String, Object?> toJson() => {${writes.map((write) => '$write,').join('\n')}};
}
''';
  }

  Kind kindOf(YamlMap schema) {
    final ref = schema[r'$ref'] as String?;
    if (ref != null) {
      final name = ref.split('/').last;
      final target = schemas[name] as YamlMap;
      if (target['type'] == 'object' && target['properties'] != null) {
        classes.add(name);
        return Kind(name, (value, at) => '$name.fromJson($value)', (value) => '$value.toJson()');
      }
      return kindOf(target);
    }

    if (schema['oneOf'] != null) {
      return Kind('Object', (value, at) => value, (value) => value);
    }

    switch (schema['type']) {
      case 'string':
        if (schema['format'] == 'date-time') {
          return Kind(
            'DateTime',
            (value, at) => "_date($value, '$at')",
            (value) => '$value.toUtc().toIso8601String()',
          );
        }
        return Kind('String', (value, at) => "_string($value, '$at')", (value) => value);
      case 'integer':
        return Kind('int', (value, at) => "_int($value, '$at')", (value) => value);
      case 'number':
        return Kind('double', (value, at) => "_double($value, '$at')", (value) => value);
      case 'boolean':
        return Kind('bool', (value, at) => "_bool($value, '$at')", (value) => value);
      case 'array':
        final item = kindOf(schema['items'] as YamlMap);
        return Kind(
          'List<${item.type}>',
          (value, at) =>
              "[for (final item in _list($value, '$at')) ${item.decode('item', '$at[]')}]",
          (value) => item.encode('item') == 'item'
              ? value
              : '[for (final item in $value) ${item.encode('item')}]',
        );
      case 'object':
        final values = schema['additionalProperties'];
        if (values is YamlMap && values['type'] == 'string') {
          return Kind(
            'Map<String, String>',
            (value, at) => "_strings($value, '$at')",
            (value) => value,
          );
        }
        return Kind(
          'Map<String, Object?>',
          (value, at) => "_object($value, '$at')",
          (value) => value,
        );
    }
    throw StateError('no Dart type for $schema');
  }

  String queryValue(Kind kind, String value) => switch (kind.type) {
    'String' => value,
    'DateTime' => '$value?.toUtc().toIso8601String()',
    _ => '$value?.toString()',
  };

  YamlMap resolveParameter(YamlMap parameter) {
    final ref = parameter[r'$ref'] as String?;
    return ref == null ? parameter : parameters[ref.split('/').last] as YamlMap;
  }

  YamlMap? jsonSchema(YamlMap withContent) {
    final content = withContent['content'] as YamlMap?;
    final json = content?['application/json'] as YamlMap?;
    return json?['schema'] as YamlMap?;
  }

  YamlMap? successSchema(YamlMap operation) {
    for (final MapEntry(key: status, value: response)
        in (operation['responses'] as YamlMap).entries) {
      if ('$status'.startsWith('2')) {
        return response is YamlMap ? jsonSchema(response) : null;
      }
    }
    return null;
  }
}

const reserved = {'default', 'in', 'is', 'new', 'null', 'switch', 'class', 'enum', 'var', 'final'};

String identifier(String wire) {
  final parts = wire.split('_');
  final camel =
      parts.first +
      parts
          .skip(1)
          .map((part) => part.isEmpty ? '' : part[0].toUpperCase() + part.substring(1))
          .join();
  return reserved.contains(camel) ? '${camel}Value' : camel;
}

String firstSentence(String text) {
  final flat = text.replaceAll(RegExp(r'\s+'), ' ').trim();
  final end = flat.indexOf('. ');
  return end == -1 ? flat : flat.substring(0, end + 1);
}

// Decoding helpers, which say which field was unreadable rather than throwing a TypeError
// from somewhere inside a constructor.
const helpers = r'''
String _string(Object? value, String at) =>
    value is String ? value : throw UnreadableException('$at is not a string');

int _int(Object? value, String at) =>
    value is num ? value.toInt() : throw UnreadableException('$at is not a number');

double _double(Object? value, String at) =>
    value is num ? value.toDouble() : throw UnreadableException('$at is not a number');

bool _bool(Object? value, String at) =>
    value is bool ? value : throw UnreadableException('$at is not a boolean');

DateTime _date(Object? value, String at) =>
    parseRouterDate(_string(value, at)) ?? (throw UnreadableException('$at is not a timestamp'));

List<Object?> _list(Object? value, String at) =>
    value is List ? value.cast<Object?>() : throw UnreadableException('$at is not a list');

Map<String, Object?> _object(Object? value, String at) =>
    value is Map ? value.cast<String, Object?>() : throw UnreadableException('$at is not an object');

Map<String, String> _strings(Object? value, String at) => {
  for (final MapEntry(:key, :value) in _object(value, at).entries) key: _string(value, '$at.$key'),
};
''';
