import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

/// A finished reply whose UI is the OpenUI [program].
class UIMessage with AIMessage {
  UIMessage(this.id, String program)
    : uiDocument = AIUIDocument.fromCustomData({
        'ai_ui': {'v': 1, 'format': 'openui', 'status': 'complete', 'content': program},
      })!;

  @override
  final String id;

  @override
  final AIUIDocument uiDocument;

  @override
  AIMessageRole get role => AIMessageRole.assistant;

  @override
  String get text => '';

  @override
  bool get isGenerating => false;

  @override
  List<AIMessagePart> get parts => const [];

  @override
  List<AIMessageAttachment> get attachments => const [];

  @override
  String? reasoningText(AIReasoningPart step) => null;
}
