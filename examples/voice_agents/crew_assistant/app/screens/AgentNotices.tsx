import { useSyncExternalStore } from 'react';
import { Pressable, StyleSheet, Text } from 'react-native';

import type { Session } from '@stream-io/stream-agents';
import { useStateStore } from 'stream-chat-expo';

import { type Chat, type ChatState, chatStore } from '../agent/conversation';
import { useAppContext } from '../contexts/AppContext';

const sessionSelector = ({ session }: ChatState) => ({ session });

const Failure = ({ session }: { session: Session }) => {
  const { retry } = useAppContext();
  const failure = useSyncExternalStore(
    session.conversation.subscribe,
    () => session.conversation.get().failure,
  );
  if (!failure) return null;
  return (
    <Pressable accessibilityRole='button' onPress={retry} style={styles.failure}>
      <Text style={styles.failureText}>Crew couldn&apos;t answer that. The model may be busy.</Text>
      <Text style={styles.retryText}>Try again</Text>
    </Pressable>
  );
};

/**
 * Why the open conversation's agent couldn't answer, above the composer, with a way to ask again.
 * A call that asks first is asked on its step in the conversation, with the approver.
 */
export const AgentNotices = ({ chat }: { chat: Chat }) => {
  const { session } = useStateStore(chatStore(chat), sessionSelector);
  return session ? <Failure session={session} /> : null;
};

const styles = StyleSheet.create({
  failure: { alignItems: 'center', paddingBottom: 8 },
  failureText: { color: '#B00020', fontSize: 14 },
  retryText: { color: '#C8102E', fontSize: 15, fontWeight: '700', paddingTop: 2 },
});
