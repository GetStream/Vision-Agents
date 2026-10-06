// What the router answers with, as it writes it.

/// A session as the router writes one, offset and nine fractional digits included.
Map<String, Object?> sessionJson({
  String id = 's1',
  String state = 'live',
  String conversationId = '',
}) => {
  'agent': 'docs',
  'agent_id': 'a1',
  'call_id': '',
  'call_type': 'agent',
  'config_id': 'c1',
  'conversation_id': conversationId,
  'created_at': '2026-09-24T09:54:56.038055123-06:00',
  'id': id,
  'instructions': 'Be brief.',
  'llm': 'gemini/gemini-3.8-flash',
  'modality': 'text',
  'mode': 'text',
  'state': state,
  'text': true,
  'user_id': 'vision-agent',
};

Map<String, Object?> responseJson(String id, {String status = 'completed'}) => {
  'id': id,
  'session_id': 's1',
  'said': 'My name is Ada.',
  'status': status,
  'created_at': '2026-09-24T15:54:56Z',
  'finished_at': '2026-09-24T15:54:57.5Z',
};

/// A page of a list, as the router pages one.
Map<String, Object?> pageJson(List<Object?> items, {String? nextCursor}) => {
  'items': items,
  'has_more': nextCursor != null,
  'next_cursor': ?nextCursor,
};
