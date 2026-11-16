import type { CrewEvent } from './contexts/AppContext';

/** One of the things the assistant helps with, as the home screen offers it. */
export type Flow = {
  id: string;
  title: string;
  subtitle: string;
  emoji: string;
} & ({ ask: string } | { event: CrewEvent });

/**
 * The flows, each started by asking the agent or by an event the restaurant's systems would send,
 * which has the agent speak first.
 */
export const FLOWS: Flow[] = [
  {
    emoji: '🕕',
    event: 'clock_in',
    id: 'clock-in',
    subtitle: 'Fit check and focus for the shift',
    title: 'Clock in',
  },
  {
    ask: "I'm not feeling well and need to call out of my next shift.",
    emoji: '🤒',
    id: 'call-out',
    subtitle: 'Policy, cover and the shift leader',
    title: 'Call out sick',
  },
  {
    ask: 'What are the uniform rules, and can I wear my own shoes?',
    emoji: '📘',
    id: 'handbook',
    subtitle: 'Answers from the handbook, with sources',
    title: 'Ask the handbook',
  },
  {
    ask: 'I think some equipment is broken and I need help!',
    emoji: '🛠️',
    id: 'equipment',
    subtitle: 'Quick fix, then a ticket if needed',
    title: 'Equipment problem',
  },
  {
    emoji: '🏖️',
    event: 'back_from_time_off',
    id: 'back',
    subtitle: 'What you missed, summarized',
    title: 'Back from time off',
  },
  {
    emoji: '👋',
    event: 'onboarding',
    id: 'onboarding',
    subtitle: 'Your first 30 days: team, menu, check-ins',
    title: "I'm new here",
  },
  {
    ask: "Let's practice for my shift leader interview. Ask me the first question.",
    emoji: '🎯',
    id: 'interview',
    subtitle: 'Mock interview with feedback',
    title: 'Interview practice',
  },
  {
    emoji: '🌙',
    event: 'clock_out',
    id: 'clock-out',
    subtitle: 'Reflect on the shift',
    title: 'Clock out',
  },
];
