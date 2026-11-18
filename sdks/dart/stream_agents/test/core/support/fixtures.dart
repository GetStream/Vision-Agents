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

/// A failure, in the envelope the router answers every one with.
Map<String, Object?> errorJson(String type, String code, String message) => {
  'error': {
    'message': message,
    'type': type,
    'code': code,
    'doc_url': 'https://getstream.io/agents/docs/api/errors/#$code',
  },
};

/// A page of a list, as the router pages one.
Map<String, Object?> pageJson(List<Object?> items, {String? nextCursor}) => {
  'items': items,
  'has_more': nextCursor != null,
  'next_cursor': ?nextCursor,
};

/// A scheduled task as the router writes one: occurrences in the task zone's offset, the rest
/// in UTC, and no run yet.
Map<String, Object?> scheduledTaskJson({
  String id = 't1',
  String state = 'active',
  String createdBy = 'agent',
}) => {
  'id': id,
  'user_id': 'ada',
  'config_id': 'c1',
  'session_id': 's1',
  'title': 'Water the plants',
  'instructions': 'Remind Ada to water the plants.',
  'schedule': {
    'at': '2026-11-19T09:00:00',
    'repeat': 'FREQ=DAILY',
    'timezone': 'Europe/Amsterdam',
    'description': 'Every day at 09:00',
  },
  'deliver_to': ['conversation', 'sms'],
  'quiet_hours': {'start': '22:00', 'end': '07:00'},
  'state': state,
  'next_run_at': '2026-11-19T09:00:00+01:00',
  'upcoming': ['2026-11-19T09:00:00+01:00', '2026-11-20T09:00:00+01:00'],
  'last_run_at': null,
  'consecutive_failures': 0,
  'created_by': createdBy,
  'custom': {'plant': 'fern'},
  'created_at': '2026-11-18T10:15:30.123456789Z',
  'updated_at': '2026-11-18T10:15:30Z',
};

/// One run of a scheduled task, delivered.
Map<String, Object?> scheduledTaskRunJson(String id) => {
  'id': id,
  'task_id': 't1',
  'scheduled_for': '2026-11-18T09:00:00+01:00',
  'trigger': 'schedule',
  'state': 'delivered',
  'delivered_to': 'conversation',
  'session_id': 's1',
  'attempts': 1,
  'started_at': '2026-11-18T08:00:01Z',
  'finished_at': '2026-11-18T08:00:04Z',
};
