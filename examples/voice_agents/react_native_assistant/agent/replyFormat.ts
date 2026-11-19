import { AgentConfig } from './config';

/**
 * How the agent answers: in Markdown, or with interactive UI in A2UI or OpenUI. This app shows
 * A2UI; the iOS and web apps show OpenUI too.
 */
export type ReplyFormat = 'markdown' | 'a2ui' | 'openui';

/** Asks swift_assistant's backend how the agent answers. */
export const replyFormat = async (): Promise<ReplyFormat> => {
  const response = await fetch(`${AgentConfig.tokenUrl}/reply-format`);
  if (!response.ok) {
    throw new Error(`The backend could not say how Juno answers (${response.status})`);
  }
  return ((await response.json()) as { format: ReplyFormat }).format;
};

// The change on its way, which the backend may apply before it answers.
let saving: Promise<unknown> = Promise.resolve();

/**
 * Settles once no change of the format is on its way, so a conversation starts with the format
 * Settings shows.
 */
export const formatSaved = () => saving.then(
  () => undefined,
  () => undefined,
);

/**
 * Has the agent answer in `format`, from its next conversation on. The choice is the agent's
 * instructions, kept by the backend, so every app answers the same way, and a conversation keeps
 * the way it started with.
 */
export const setReplyFormat = (format: ReplyFormat) => {
  const request = (async () => {
    const response = await fetch(`${AgentConfig.tokenUrl}/reply-format`, {
      body: JSON.stringify({ format }),
      headers: { 'Content-Type': 'application/json' },
      method: 'PUT',
    });
    if (!response.ok) {
      throw new Error(`The backend could not change how Juno answers (${response.status})`);
    }
  })();
  saving = request;
  return request;
};
