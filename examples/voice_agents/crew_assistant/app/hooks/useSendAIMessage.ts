import { useStableCallback } from 'stream-chat-expo';

import type { ComposerViewProps } from 'stream-chat-expo/ai-components';

import { useAppContext } from '../contexts/AppContext';

export type AIMessage = Parameters<NonNullable<ComposerViewProps['onSendMessage']>>[0];

/**
 * Asks the agent through the open conversation, as the composer would: the question shows at once,
 * and the router writes it and its reply into the conversation's channel. Used by voice mode.
 */
export const useSendAIMessage = () => {
  const { conversation } = useAppContext();

  return useStableCallback((message: AIMessage) => {
    conversation?.send(message);
    return Promise.resolve();
  });
};
