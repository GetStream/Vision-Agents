import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

/// A reply whose UI is the OpenUI [program], finished unless [isStreaming].
class UIMessage with AIMessage {
  UIMessage(this.id, String program, {this.isStreaming = false})
    : uiDocument = AIUIDocument.fromCustomData({
        'ai_ui': {
          'v': 1,
          'format': 'openui',
          'status': isStreaming ? 'streaming' : 'complete',
          'content': program,
        },
      })!;

  @override
  final String id;

  @override
  final AIUIDocument uiDocument;

  @override
  AIMessageRole get role => AIMessageRole.assistant;

  @override
  String get text => '';

  final bool isStreaming;

  @override
  bool get isGenerating => isStreaming;

  @override
  List<AIMessagePart> get parts => const [];

  @override
  List<AIMessageAttachment> get attachments => const [];

  @override
  String? reasoningText(AIReasoningPart step) => null;
}
