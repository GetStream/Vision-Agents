import type { AIChatOption } from 'stream-chat-expo/ai-components';

/** Asks with more time to reason. */
export const THINK_HARDER: AIChatOption = {
  description: 'More time to reason before answering',
  id: 'think_harder',
  shortTitle: 'Think',
  title: 'Think harder',
};

export const chatOptions = [THINK_HARDER];

/** Starters for an empty conversation. */
export const STARTERS = [
  "What's on my calendar today?",
  'Remind me to call mum at 6pm',
  'Plan a weekend in Lisbon',
  'What should I cook tonight?',
];
