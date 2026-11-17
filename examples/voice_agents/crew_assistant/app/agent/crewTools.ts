import type { ToolApproval } from '@stream-io/stream-agents';

import { type AIClientTool, completedToolResult } from 'stream-chat-expo/ai-components';

import { crewBackend } from './agents';

type Schema = Record<string, unknown>;

const object = (properties: Record<string, Schema>, required: string[] = []): Schema => ({
  properties,
  required,
  type: 'object',
});
const text = (description: string): Schema => ({ description, type: 'string' });

const tool = (
  name: string,
  description: string,
  inputSchema: Schema,
  run: (input: Record<string, unknown>) => Promise<unknown>,
  summary: string,
  readOnly = true,
): AIClientTool => ({
  definition: {
    annotations: { readOnlyHint: readOnly },
    description,
    inputSchema,
    name,
  },
  run: async (call) =>
    completedToolResult(await run((call.arguments ?? {}) as Record<string, unknown>), summary),
});

const localContext: AIClientTool = {
  definition: {
    annotations: { readOnlyHint: true },
    description:
      "The person's current date, local time, timezone and language. Call it before answering anything that depends on today or the time.",
    inputSchema: { properties: {}, type: 'object' },
    name: 'get_local_context',
  },
  run: () => {
    const now = new Date();
    const { locale, timeZone } = Intl.DateTimeFormat().resolvedOptions();
    return completedToolResult(
      { language: locale, local: now.toString(), timeZone, utc: now.toISOString() },
      'Shared the local time',
    );
  },
};

/**
 * The phone's tools: its clock, and the three actions the person says yes to here before they
 * run. The restaurant's other tools run on the crew backend, which hosts them on the router.
 */
export const CREW_TOOLS: AIClientTool[] = [
  localContext,
  tool(
    'request_shift_swap',
    'Asks a teammate to take one of the person’s shifts. The phone asks the person first.',
    object(
      {
        note: text('A short note to the teammate'),
        shift_id: text('The shift to give away'),
        teammate_id: text('The teammate, from find_coverage'),
      },
      ['shift_id', 'teammate_id'],
    ),
    (body) => crewBackend('crew/swaps', { body }),
    'Asked for a swap',
    false,
  ),
  tool(
    'create_support_ticket',
    'Drafts a repair ticket for the owner to review and submit. The phone asks the person first.',
    object(
      {
        asset: text('The asset id'),
        details: text('What they saw and already tried'),
        priority: { enum: ['low', 'normal', 'urgent'], type: 'string' },
        summary: text('The problem, in a short line'),
      },
      ['summary', 'details'],
    ),
    (body) => crewBackend('crew/tickets', { body }),
    'Drafted a ticket',
    false,
  ),
  tool(
    'book_check_in',
    'Books a 1-on-1 check-in with a leader or trainer. The phone asks the person first.',
    object(
      {
        topic: text('What it is about'),
        when: text('When, such as "Thursday 14:00"'),
        with: text('Who, by id or name, from get_team'),
      },
      ['with', 'when'],
    ),
    (body) => crewBackend('crew/check-ins', { body }),
    'Booked a check-in',
    false,
  ),
];

/**
 * What a call says it is doing on the reply's step, by tool: the phone's own tools declare theirs,
 * and the conversation gives the backend's theirs.
 */
export const DISPLAY_TITLES: Record<string, string> = {
  book_check_in: 'Booking a check-in',
  calcom__call_tool: 'Using your Cal.com',
  calcom__list_tools: 'Opening your Cal.com',
  check_callout: 'Checking the call-out policy',
  create_support_ticket: 'Drafting a repair ticket',
  find_coverage: 'Finding who can cover',
  get_equipment_guide: 'Looking up the quick fix',
  get_local_context: "Checking your phone's time",
  get_menu: 'Reading the menu',
  get_my_schedule: 'Reading your schedule',
  get_team: 'Looking up the team',
  get_updates_since: 'Reading the announcements',
  notify_shift_leader: 'Telling the shift leader',
  request_shift_swap: 'Asking for a swap',
  save_shift_reflection: 'Saving your reflection',
  search_handbook: 'Searching the handbook',
};

/** What the person is asked before a call runs, by tool. */
export const APPROVALS: Record<string, ToolApproval> = {
  book_check_in: { allowTitle: 'Book it', title: 'Book this check-in?' },
  create_support_ticket: {
    allowTitle: 'Send for review',
    message: 'The owner reviews it before it goes to repairs.',
    title: 'Draft this repair ticket?',
  },
  request_shift_swap: {
    allowTitle: 'Send request',
    message: 'Your teammate and the shift leader see it.',
    title: 'Ask a teammate to take your shift?',
  },
};
