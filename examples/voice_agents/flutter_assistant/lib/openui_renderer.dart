import 'dart:math' as math;

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'brand_theme.dart';

/// Native Flutter renderer for OpenUI Lang 1.0 (`openui`), supporting incremental
/// line-by-line streaming, forward-reference hoisting, `$variable` state bindings, built-in
/// layout/content/chart/table/form primitives, and app `AIUIComponent` portable views.
final class OpenUIRenderer extends AIUIRenderer {
  OpenUIRenderer({this.onError});

  static const String fence = 'openui-lang';

  final void Function(Object error, String messageId)? onError;

  @override
  AIUIFormat get format => AIUIFormat.openUI;

  @override
  Map<String, Object?> capabilities(
    List<AIUIComponent<Object?>> components,
    List<AIToolDefinition> tools,
  ) => {
    'fence': fence,
    'components': [for (final c in components) c.definition.name],
  };

  @override
  AIUIEngine createEngine(AIUIHosting host) => _OpenUIEngine(host, onError);
}

final class _OpenUIEngine extends AIUIEngine {
  _OpenUIEngine(this.host, this.onError);

  final AIUIHosting host;
  final void Function(Object error, String messageId)? onError;
  final _programs = <String, _ParsedOpenUI>{};
  final _customByPascal = <String, AIUIComponent<Object?>>{};

  AIUIComponent<Object?>? _lookupCustom(String pascal) {
    if (_customByPascal.isEmpty && host.components.isNotEmpty) {
      for (final comp in host.components) {
        _customByPascal[_pascalName(comp.name)] = comp;
      }
    }
    return _customByPascal[pascal];
  }

  @override
  void receive(AIUIDocument document, String messageId) {
    final text = document.content is String ? document.content as String : '';
    _programs[messageId] = _ParsedOpenUI.parse(text, isStreaming: document.isStreaming);
  }

  @override
  void remove(String messageId) {
    _programs.remove(messageId);
  }

  @override
  Widget? view(AIUIDocument document, String messageId) {
    final text = document.content is String ? document.content as String : '';
    final parsed = _programs[messageId] ??= _ParsedOpenUI.parse(
      text,
      isStreaming: document.isStreaming,
    );
    if (!parsed.hasRoot) return null;
    return _OpenUISurface(
      key: ValueKey('openui-$messageId'),
      host: host,
      messageId: messageId,
      program: parsed,
      lookupCustom: _lookupCustom,
      onError: onError,
    );
  }

  @override
  void dispose() {
    _programs.clear();
  }
}

class _OpenUISurface extends StatefulWidget {
  const _OpenUISurface({
    super.key,
    required this.host,
    required this.messageId,
    required this.program,
    required this.lookupCustom,
    this.onError,
  });

  final AIUIHosting host;
  final String messageId;
  final _ParsedOpenUI program;
  final AIUIComponent<Object?>? Function(String pascal) lookupCustom;
  final void Function(Object error, String messageId)? onError;

  @override
  State<_OpenUISurface> createState() => _OpenUISurfaceState();
}

class _OpenUISurfaceState extends State<_OpenUISurface> {
  late Map<String, Object?> _vars = _initialVars();
  final _tabs = <String, int>{};

  Map<String, Object?> _initialVars() {
    final restored = widget.host.state(widget.messageId);
    if (restored is Map<String, Object?>) {
      return {...widget.program.initialState, ...restored};
    }
    return {...widget.program.initialState};
  }

  @override
  void didUpdateWidget(_OpenUISurface oldWidget) {
    super.didUpdateWidget(oldWidget);
    for (final entry in widget.program.initialState.entries) {
      _vars.putIfAbsent(entry.key, () => entry.value);
    }
  }

  void _setVar(String key, Object? value) {
    setState(() {
      _vars = {..._vars, key: value};
    });
    widget.host.setState(widget.messageId, _vars);
  }

  void _emitAction(String name, String label, Map<String, Object?> payload) {
    if (widget.program.isStreaming) return;
    widget.host.send(
      AIUIAction(
        messageId: widget.messageId,
        format: AIUIFormat.openUI,
        name: name,
        label: label,
        payload: payload,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final rootExpr = widget.program.statements['root'];
    if (rootExpr == null) return const SizedBox.shrink();
    final built = _buildExpr(context, rootExpr, 'root');
    return built ?? const SizedBox.shrink();
  }

  Widget? _buildExpr(BuildContext context, _Expr expr, String nodeKey) {
    final resolved = _deref(expr, <String>{});
    if (resolved is! _CallExpr) return null;
    final custom = widget.lookupCustom(resolved.name);
    if (custom != null) {
      return _buildCustomComponent(context, custom, resolved.args);
    }

    final colors = context.streamColorScheme;
    final textTheme = context.streamTextTheme;
    final brand = BrandThemeData.of(context);

    switch (resolved.name) {
      case 'Card':
        final children = _buildChildList(context, resolved.argAt(0), '$nodeKey.c');
        if (children.isEmpty && widget.program.isStreaming) return null;
        return DecoratedBox(
          decoration: BoxDecoration(
            color: colors.backgroundSurfaceCard,
            borderRadius: BorderRadius.circular(brand.cornerRadius),
            border: Border.all(color: colors.borderDefault),
          ),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              spacing: 12,
              children: children,
            ),
          ),
        );

      case 'CardHeader' || 'InlineHeader':
        final title = _evalString(resolved.argAt(0));
        final subtitle = _evalString(resolved.argAt(1));
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 2,
          children: [
            if (title.isNotEmpty)
              Text(
                title,
                style: (resolved.name == 'CardHeader' ? textTheme.headingMd : textTheme.headingSm)
                    .copyWith(color: colors.textPrimary),
              ),
            if (subtitle.isNotEmpty)
              Text(subtitle, style: textTheme.captionDefault.copyWith(color: colors.textSecondary)),
          ],
        );

      case 'TextContent' || 'MarkDownRenderer' || 'Label':
        final body = _evalString(resolved.argAt(0));
        if (body.isEmpty) return null;
        return AIMarkdownBody(data: body);

      case 'Callout' || 'TextCallout':
        final variant = _evalString(resolved.argAt(0));
        final title = _evalString(resolved.argAt(1));
        final desc = _evalString(resolved.argAt(2));
        final toneColor = switch (variant) {
          'warning' => colors.accentWarning,
          'error' || 'danger' => colors.accentError,
          'success' => colors.accentSuccess,
          _ => brand.accent,
        };
        return DecoratedBox(
          decoration: BoxDecoration(
            color: toneColor.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular((brand.cornerRadius * 0.75).clamp(6, 14)),
            border: Border.all(color: toneColor.withValues(alpha: 0.35)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              spacing: 4,
              children: [
                if (title.isNotEmpty)
                  Text(title, style: textTheme.captionEmphasis.copyWith(color: colors.textPrimary)),
                if (desc.isNotEmpty)
                  Text(desc, style: textTheme.captionDefault.copyWith(color: colors.textSecondary)),
              ],
            ),
          ),
        );

      case 'Separator':
        return Divider(height: 1, color: colors.borderDefault);

      case 'TagBlock':
        final tags = _evalList(resolved.argAt(0)).map((e) => '$e').toList();
        if (tags.isEmpty) return null;
        return Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final tag in tags)
              Chip(
                label: Text(tag, style: textTheme.metadataDefault),
                visualDensity: VisualDensity.compact,
              ),
          ],
        );

      case 'Steps':
        final items = _resolveCallList(resolved.argAt(0));
        if (items.isEmpty) return null;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 8,
          children: [
            for (var i = 0; i < items.length; i++)
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                spacing: 10,
                children: [
                  CircleAvatar(
                    radius: 11,
                    backgroundColor: brand.accent.withValues(alpha: 0.15),
                    child: Text(
                      '${i + 1}',
                      style: textTheme.metadataDefault.copyWith(color: brand.accent),
                    ),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      spacing: 2,
                      children: [
                        Text(
                          _evalString(items[i].argAt(0)),
                          style: textTheme.captionEmphasis.copyWith(color: colors.textPrimary),
                        ),
                        if (_evalString(items[i].argAt(1)).isNotEmpty)
                          Text(
                            _evalString(items[i].argAt(1)),
                            style: textTheme.captionDefault.copyWith(color: colors.textSecondary),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
          ],
        );

      case 'Tabs':
        final tabs = _resolveCallList(resolved.argAt(0));
        if (tabs.isEmpty) return null;
        final selected = (_tabs[nodeKey] ?? 0).clamp(0, tabs.length - 1);
        final activeTab = tabs[selected];
        final panelChildren = _buildChildList(context, activeTab.argAt(2), '$nodeKey.t$selected');
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          spacing: 10,
          children: [
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                spacing: 8,
                children: [
                  for (var i = 0; i < tabs.length; i++)
                    ChoiceChip(
                      label: Text(_evalString(tabs[i].argAt(1))),
                      selected: i == selected,
                      selectedColor: brand.accent.withValues(alpha: 0.16),
                      onSelected: (_) => setState(() => _tabs[nodeKey] = i),
                    ),
                ],
              ),
            ),
            ...panelChildren,
          ],
        );

      case 'Form':
        final formName = _evalString(resolved.argAt(0));
        final buttonsWidget = _buildExpr(
          context,
          resolved.argAt(1) ?? const _NullExpr(),
          '$nodeKey.btns',
        );
        final fields = _buildChildList(context, resolved.argAt(2), '$nodeKey.fields');
        return DecoratedBox(
          decoration: BoxDecoration(
            color: colors.backgroundSurface,
            borderRadius: BorderRadius.circular((brand.cornerRadius * 0.85).clamp(6, 16)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              spacing: 10,
              children: [
                ...fields,
                if (buttonsWidget != null) _FormContext(formName: formName, child: buttonsWidget),
              ],
            ),
          ),
        );

      case 'FormControl':
        final label = _evalString(resolved.argAt(0));
        final inputExpr = resolved.argAt(1);
        final hint = _evalString(resolved.argAt(2));
        final control = inputExpr == null ? null : _buildFormInput(context, inputExpr);
        if (control == null) return null;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 6,
          children: [
            if (label.isNotEmpty)
              Text(label, style: textTheme.captionEmphasis.copyWith(color: colors.textPrimary)),
            control,
            if (hint.isNotEmpty)
              Text(hint, style: textTheme.metadataDefault.copyWith(color: colors.textTertiary)),
          ],
        );

      case 'Buttons':
        final buttons = _resolveCallList(resolved.argAt(0));
        return Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (var i = 0; i < buttons.length; i++)
              if (_buildButton(context, buttons[i]) case final Widget btn) btn,
          ],
        );

      case 'Button':
        return _buildButton(context, resolved);

      case 'FollowUpBlock':
        final items = _resolveCallList(resolved.argAt(0));
        if (items.isEmpty) return null;
        return Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final item in items)
              ActionChip(
                label: Text(_evalString(item.argAt(0))),
                onPressed: widget.program.isStreaming
                    ? null
                    : () {
                        final text = _evalString(item.argAt(0));
                        _emitAction('continue_conversation', text, {'message': text});
                      },
              ),
          ],
        );

      case 'PieChart':
        final labels = _evalList(resolved.argAt(0)).map((e) => '$e').toList();
        final values = _evalList(
          resolved.argAt(1),
        ).map((e) => e is num ? e.toDouble() : double.tryParse('$e') ?? 0).toList();
        if (labels.isEmpty || values.isEmpty) return null;
        final total = values.fold<double>(0, (a, b) => a + math.max(0, b));
        final firstPct = total > 0 ? ((values.first / total) * 100).round() : 0;
        return Row(
          spacing: 12,
          children: [
            for (var i = 0; i < math.min(labels.length, values.length); i++)
              Expanded(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: colors.backgroundSurface,
                    borderRadius: BorderRadius.circular(brand.cornerRadius * 0.75),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(10),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          labels[i],
                          style: textTheme.metadataDefault.copyWith(color: colors.textSecondary),
                        ),
                        Text(
                          i == 0 ? '$firstPct%' : '${values[i]}',
                          style: textTheme.headingSm.copyWith(color: brand.accent),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        );

      default:
        return null;
    }
  }

  Widget? _buildFormInput(BuildContext context, _Expr expr) {
    final resolved = _deref(expr, <String>{});
    if (resolved is! _CallExpr) return null;
    final brand = BrandThemeData.of(context);

    switch (resolved.name) {
      case 'Chips':
        final fieldName = _evalString(resolved.argAt(0));
        final mode = _evalString(resolved.argAt(1));
        final items = _resolveCallList(resolved.argAt(2));
        final defaultVal = _evalValue(resolved.argAt(4));
        final current = _vars['field:$fieldName'] ?? defaultVal;
        final isMulti = mode == 'multiple';
        return Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final chip in items)
              Builder(
                builder: (context) {
                  final val = _evalString(chip.argAt(0));
                  final label = _evalString(chip.argAt(1));
                  final selected = isMulti
                      ? (current is List && current.contains(val))
                      : current == val;
                  return FilterChip(
                    label: Text(label),
                    selected: selected,
                    selectedColor: brand.accent.withValues(alpha: 0.18),
                    onSelected: widget.program.isStreaming
                        ? null
                        : (picked) {
                            if (isMulti) {
                              final next = <String>[
                                if (current is List) ...current.map((e) => '$e'),
                              ];
                              if (picked) {
                                if (!next.contains(val)) next.add(val);
                              } else {
                                next.remove(val);
                              }
                              _setVar('field:$fieldName', next);
                            } else {
                              _setVar('field:$fieldName', val);
                            }
                          },
                  );
                },
              ),
          ],
        );

      case 'Slider':
        final fieldName = _evalString(resolved.argAt(0));
        final min = _evalNumber(resolved.argAt(2), 0);
        final max = _evalNumber(resolved.argAt(3), 10);
        final step = _evalNumber(resolved.argAt(4), 1);
        final defList = _evalList(resolved.argAt(5));
        final defVal = defList.isNotEmpty && defList.first is num
            ? (defList.first! as num).toDouble()
            : min;
        final current =
            (_vars['field:$fieldName'] is num
                    ? (_vars['field:$fieldName']! as num).toDouble()
                    : defVal)
                .clamp(min, max);
        final divisions = step > 0 ? ((max - min) / step).round().clamp(1, 100) : null;
        return Slider(
          value: current,
          min: min,
          max: max,
          divisions: divisions,
          activeColor: brand.accent,
          label: '${current.round()}',
          onChanged: widget.program.isStreaming ? null : (v) => _setVar('field:$fieldName', v),
        );

      default:
        return null;
    }
  }

  Widget? _buildButton(BuildContext context, _CallExpr button) {
    final label = _evalString(button.argAt(0));
    if (label.isEmpty) return null;
    final actionExpr = button.argAt(1);
    final variant = _evalString(button.argAt(2));
    final brand = BrandThemeData.of(context);
    final formName = context.dependOnInheritedWidgetOfExactType<_FormContext>()?.formName;

    void handleTap() {
      final formValues = <String, Object?>{
        for (final entry in _vars.entries)
          if (entry.key.startsWith('field:')) entry.key.substring(6): entry.value,
        for (final entry in _vars.entries)
          if (entry.key.startsWith(r'$')) entry.key.substring(1): entry.value,
      };
      final action = _resolveAction(actionExpr);
      if (action.url != null) {
        final uri = Uri.tryParse(action.url!);
        if (uri != null) widget.host.open(uri);
        return;
      }
      _emitAction(action.name, label, {
        if (formName != null && formName.isNotEmpty) 'form': formName,
        if (formValues.isNotEmpty) 'values': formValues,
        if (action.context != null) 'context': action.context,
        if (action.message != null && action.message != label) 'message': action.message,
      });
    }

    if (variant == 'primary') {
      return FilledButton(
        style: FilledButton.styleFrom(backgroundColor: brand.accent),
        onPressed: widget.program.isStreaming ? null : handleTap,
        child: Text(label),
      );
    }
    return OutlinedButton(
      onPressed: widget.program.isStreaming ? null : handleTap,
      child: Text(label),
    );
  }

  ({String name, String? message, Object? context, String? url}) _resolveAction(_Expr? expr) {
    if (expr == null) {
      return (name: 'continue_conversation', message: null, context: null, url: null);
    }
    final resolved = _deref(expr, <String>{});
    if (resolved is _CallExpr) {
      if (resolved.name == 'Action') {
        final steps = _resolveCallList(resolved.argAt(0));
        for (final step in steps) {
          final res = _resolveAction(step);
          if (res.url != null || res.message != null) return res;
        }
      }
      if (resolved.name == '@OpenUrl') {
        return (
          name: 'open_url',
          message: null,
          context: null,
          url: _evalString(resolved.argAt(0)),
        );
      }
      if (resolved.name == '@ToAssistant') {
        return (
          name: 'continue_conversation',
          message: _evalString(resolved.argAt(0)),
          context: _evalValue(resolved.argAt(1)),
          url: null,
        );
      }
    }
    return (name: 'continue_conversation', message: null, context: null, url: null);
  }

  Widget? _buildCustomComponent(
    BuildContext context,
    AIUIComponent<Object?> component,
    List<_Expr> positionalArgs,
  ) {
    final params = component.definition.parameters;
    final map = <String, Object?>{};
    for (var i = 0; i < math.min(params.length, positionalArgs.length); i++) {
      final value = _evalValue(positionalArgs[i]);
      if (value != null) {
        map[params[i].name] = value;
      }
    }
    return component.view(
      map,
      AIUIComponentContext(
        messageId: widget.messageId,
        isStreaming: widget.program.isStreaming,
        host: widget.host,
        emit: (event, label, payload) async {
          if (widget.program.isStreaming) return;
          await widget.host.send(
            AIUIAction(
              messageId: widget.messageId,
              format: AIUIFormat.openUI,
              name: event,
              label: label,
              payload: {...payload, 'component': component.name},
            ),
          );
        },
      ),
    );
  }

  List<Widget> _buildChildList(BuildContext context, _Expr? expr, String prefix) {
    if (expr == null) return const [];
    final resolved = _deref(expr, <String>{});
    if (resolved is! _ArrayExpr) return const [];
    final out = <Widget>[];
    for (var i = 0; i < resolved.items.length; i++) {
      final w = _buildExpr(context, resolved.items[i], '$prefix.$i');
      if (w != null) out.add(w);
    }
    return out;
  }

  List<_CallExpr> _resolveCallList(_Expr? expr) {
    if (expr == null) return const [];
    final resolved = _deref(expr, <String>{});
    if (resolved is! _ArrayExpr) return const [];
    return [
      for (final item in resolved.items)
        if (_deref(item, <String>{}) case final _CallExpr call) call,
    ];
  }

  _Expr _deref(_Expr expr, Set<String> seen) {
    if (expr is _IdentExpr && !expr.name.startsWith(r'$')) {
      if (seen.contains(expr.name)) return const _NullExpr();
      final target = widget.program.statements[expr.name];
      if (target == null) return const _NullExpr();
      return _deref(target, {...seen, expr.name});
    }
    return expr;
  }

  Object? _evalValue(_Expr? expr) {
    if (expr == null) return null;
    final resolved = _deref(expr, <String>{});
    return switch (resolved) {
      _LiteralExpr(:final value) => value,
      _NullExpr() => null,
      _IdentExpr(:final name) when name.startsWith(r'$') => _vars[name],
      _IdentExpr() => null,
      _ArrayExpr(:final items) => [for (final item in items) _evalValue(item)],
      _ObjectExpr(:final entries) => {
        for (final MapEntry(:key, :value) in entries.entries) key: _evalValue(value),
      },
      _CallExpr() => null,
    };
  }

  String _evalString(_Expr? expr) => switch (_evalValue(expr)) {
    final String s => s,
    final num n => '$n',
    _ => '',
  };

  double _evalNumber(_Expr? expr, double fallback) => switch (_evalValue(expr)) {
    final num n => n.toDouble(),
    final String s => double.tryParse(s) ?? fallback,
    _ => fallback,
  };

  List<Object?> _evalList(_Expr? expr) => switch (_evalValue(expr)) {
    final List<Object?> l => l,
    _ => const [],
  };
}

class _FormContext extends InheritedWidget {
  const _FormContext({required this.formName, required super.child});

  final String formName;

  @override
  bool updateShouldNotify(_FormContext oldWidget) => oldWidget.formName != formName;
}

final class _ParsedOpenUI {
  const _ParsedOpenUI({
    required this.statements,
    required this.initialState,
    required this.isStreaming,
  });

  final Map<String, _Expr> statements;
  final Map<String, Object?> initialState;
  final bool isStreaming;

  bool get hasRoot => statements['root'] is _CallExpr;

  static _ParsedOpenUI parse(String raw, {required bool isStreaming}) {
    final stripped = _stripFence(raw);
    final lines = _splitStatements(stripped, autoCloseTail: isStreaming);
    final statements = <String, _Expr>{};
    final initialState = <String, Object?>{};

    for (final stmt in lines) {
      final eq = _findTopEquals(stmt);
      if (eq <= 0) continue;
      final lhs = stmt.substring(0, eq).trim();
      final rhs = stmt.substring(eq + 1).trim();
      if (lhs.isEmpty || rhs.isEmpty) continue;
      final parsed = _ExprParser(rhs).parse();
      if (parsed == null) continue;
      if (lhs.startsWith(r'$')) {
        initialState[lhs] = _staticEval(parsed);
      } else {
        statements[lhs] = parsed;
      }
    }
    return _ParsedOpenUI(
      statements: statements,
      initialState: initialState,
      isStreaming: isStreaming,
    );
  }

  static Object? _staticEval(_Expr expr) => switch (expr) {
    _LiteralExpr(:final value) => value,
    _NullExpr() => null,
    _ArrayExpr(:final items) => [for (final i in items) _staticEval(i)],
    _ObjectExpr(:final entries) => {
      for (final MapEntry(:key, :value) in entries.entries) key: _staticEval(value),
    },
    _ => null,
  };

  static String _stripFence(String text) {
    final trimmed = text.trim();
    if (trimmed.startsWith('```')) {
      final firstNl = trimmed.indexOf('\n');
      if (firstNl < 0) return '';
      var body = trimmed.substring(firstNl + 1);
      if (body.endsWith('```')) {
        body = body.substring(0, body.length - 3);
      }
      return body;
    }
    return text;
  }

  static List<String> _splitStatements(String source, {required bool autoCloseTail}) {
    final stmts = <String>[];
    final current = StringBuffer();
    var quote = '';
    var escaped = false;
    final stack = <String>[];

    for (final line in source.split('\n')) {
      if (quote.isEmpty && stack.isEmpty) {
        final trimmed = line.trim();
        if (trimmed.isEmpty || trimmed.startsWith('//') || trimmed.startsWith('#')) {
          continue;
        }
      }
      if (current.isNotEmpty) current.write('\n');
      current.write(line);

      for (var i = 0; i < line.length; i++) {
        final ch = line[i];
        if (escaped) {
          escaped = false;
          continue;
        }
        if (quote.isNotEmpty) {
          if (ch == r'\') {
            escaped = true;
          } else if (ch == quote) {
            quote = '';
          }
          continue;
        }
        if (ch == '/' && i + 1 < line.length && line[i + 1] == '/') break;
        if (ch == '"' || ch == "'") {
          quote = ch;
        } else if (ch == '(') {
          stack.add(')');
        } else if (ch == '[') {
          stack.add(']');
        } else if (ch == '{') {
          stack.add('}');
        } else if ((ch == ')' || ch == ']' || ch == '}') && stack.isNotEmpty) {
          stack.removeLast();
        }
      }

      if (quote.isEmpty && stack.isEmpty) {
        final done = current.toString().trim();
        if (done.isNotEmpty) stmts.add(done);
        current.clear();
      }
    }

    if (current.isNotEmpty && autoCloseTail) {
      if (quote.isNotEmpty) current.write(quote);
      for (final closer in stack.reversed) {
        current.write(closer);
      }
      final healed = current.toString().trim();
      if (healed.isNotEmpty) stmts.add(healed);
    }
    return stmts;
  }

  static int _findTopEquals(String stmt) {
    for (var i = 0; i < stmt.length; i++) {
      if (stmt[i] != '=') continue;
      final prev = i > 0 ? stmt[i - 1] : '';
      if (prev == '!' || prev == '<' || prev == '>' || prev == '=') continue;
      if (i + 1 < stmt.length && stmt[i + 1] == '=') {
        i++;
        continue;
      }
      return i;
    }
    return -1;
  }
}

sealed class _Expr {
  const _Expr();
}

final class _NullExpr extends _Expr {
  const _NullExpr();
}

final class _LiteralExpr extends _Expr {
  const _LiteralExpr(this.value);
  final Object value;
}

final class _IdentExpr extends _Expr {
  const _IdentExpr(this.name);
  final String name;
}

final class _ArrayExpr extends _Expr {
  const _ArrayExpr(this.items);
  final List<_Expr> items;
}

final class _ObjectExpr extends _Expr {
  const _ObjectExpr(this.entries);
  final Map<String, _Expr> entries;
}

final class _CallExpr extends _Expr {
  const _CallExpr(this.name, this.args);
  final String name;
  final List<_Expr> args;

  _Expr? argAt(int index) => index < args.length ? args[index] : null;
}

final class _ExprParser {
  _ExprParser(this.src);

  final String src;
  int pos = 0;

  void _skipWs() {
    while (pos < src.length) {
      final ch = src[pos];
      if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
        pos++;
      } else {
        break;
      }
    }
  }

  _Expr? parse() {
    _skipWs();
    if (pos >= src.length) return null;
    final ch = src[pos];
    if (ch == '"' || ch == "'") return _parseString();
    if (ch == '[') return _parseArray();
    if (ch == '{') return _parseObject();
    if (ch == '-' || (ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57)) {
      return _parseNumber();
    }
    return _parseIdentOrCall();
  }

  _Expr? _parseString() {
    final quote = src[pos++];
    final buf = StringBuffer();
    while (pos < src.length) {
      final ch = src[pos++];
      if (ch == r'\' && pos < src.length) {
        final esc = src[pos++];
        switch (esc) {
          case 'n':
            buf.write('\n');
          case 't':
            buf.write('\t');
          case 'r':
            buf.write('\r');
          default:
            buf.write(esc);
        }
      } else if (ch == quote) {
        break;
      } else {
        buf.write(ch);
      }
    }
    return _LiteralExpr(buf.toString());
  }

  _Expr _parseNumber() {
    final start = pos;
    if (src[pos] == '-') pos++;
    while (pos < src.length) {
      final c = src[pos];
      if ((c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57) || c == '.') {
        pos++;
      } else {
        break;
      }
    }
    final text = src.substring(start, pos);
    final numVal = int.tryParse(text) ?? double.tryParse(text) ?? 0;
    return _LiteralExpr(numVal);
  }

  _Expr _parseArray() {
    pos++; // '['
    final items = <_Expr>[];
    while (pos < src.length) {
      _skipWs();
      if (pos >= src.length || src[pos] == ']') {
        if (pos < src.length) pos++;
        break;
      }
      final item = parse();
      if (item != null) items.add(item);
      _skipWs();
      if (pos < src.length && src[pos] == ',') pos++;
    }
    return _ArrayExpr(items);
  }

  _Expr _parseObject() {
    pos++; // '{'
    final entries = <String, _Expr>{};
    while (pos < src.length) {
      _skipWs();
      if (pos >= src.length || src[pos] == '}') {
        if (pos < src.length) pos++;
        break;
      }
      String key;
      if (src[pos] == '"' || src[pos] == "'") {
        key = (_parseString() as _LiteralExpr?)?.value as String? ?? '';
      } else {
        final start = pos;
        while (pos < src.length &&
            src[pos] != ':' &&
            src[pos] != '}' &&
            !RegExp(r'\s').hasMatch(src[pos])) {
          pos++;
        }
        key = src.substring(start, pos);
      }
      _skipWs();
      if (pos < src.length && src[pos] == ':') pos++;
      final val = parse() ?? const _NullExpr();
      if (key.isNotEmpty) entries[key] = val;
      _skipWs();
      if (pos < src.length && src[pos] == ',') pos++;
    }
    return _ObjectExpr(entries);
  }

  _Expr? _parseIdentOrCall() {
    final start = pos;
    while (pos < src.length) {
      final c = src[pos];
      if (RegExp(r'[A-Za-z0-9_$@.]').hasMatch(c)) {
        pos++;
      } else {
        break;
      }
    }
    if (pos == start) {
      pos++;
      return null;
    }
    final token = src.substring(start, pos);
    _skipWs();
    if (pos < src.length && src[pos] == '(') {
      pos++; // '('
      final args = <_Expr>[];
      while (pos < src.length) {
        _skipWs();
        if (pos >= src.length || src[pos] == ')') {
          if (pos < src.length) pos++;
          break;
        }
        final arg = parse();
        if (arg != null) args.add(arg);
        _skipWs();
        if (pos < src.length && src[pos] == ',') pos++;
      }
      return _CallExpr(token, args);
    }
    switch (token) {
      case 'true':
        return const _LiteralExpr(true);
      case 'false':
        return const _LiteralExpr(false);
      case 'null':
        return const _NullExpr();
      default:
        return _IdentExpr(token);
    }
  }
}

String _pascalName(String snake) => snake
    .split('_')
    .where((part) => part.isNotEmpty)
    .map((part) => '${part[0].toUpperCase()}${part.substring(1)}')
    .join();
