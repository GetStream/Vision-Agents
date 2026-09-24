import 'package:flutter/material.dart';

/// Where you type.
///
/// Sending is a callback rather than a session, so the same field works for a conversation,
/// a search box or anything else the host wants it for.
class Composer extends StatefulWidget {
  const Composer({super.key, required this.send, this.prompt = 'Message', this.enabled = true});

  final Future<void> Function(String text) send;
  final String prompt;
  final bool enabled;

  @override
  State<Composer> createState() => _ComposerState();
}

class _ComposerState extends State<Composer> {
  final _text = TextEditingController();

  @override
  void initState() {
    super.initState();
    _text.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  bool get _canSend => widget.enabled && _text.text.trim().isNotEmpty;

  void _submit() {
    if (!_canSend) {
      return;
    }
    final sending = _text.text;
    _text.clear();
    widget.send(sending);
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _text,
            enabled: widget.enabled,
            minLines: 1,
            maxLines: 5,
            textInputAction: TextInputAction.send,
            onSubmitted: (_) => _submit(),
            decoration: InputDecoration(
              hintText: widget.prompt,
              isDense: true,
              filled: true,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: BorderSide.none,
              ),
            ),
          ),
        ),
        const SizedBox(width: 8),
        IconButton.filled(
          tooltip: 'Send',
          onPressed: _canSend ? _submit : null,
          icon: const Icon(Icons.arrow_upward),
        ),
      ],
    );
  }
}
