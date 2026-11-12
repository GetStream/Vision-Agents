import { useSyncExternalStore } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { Session, ToolApprovalRequest } from '@stream-io/stream-agents';
import { useStateStore } from 'stream-chat-expo';
import { AIToolApprovalCard } from 'stream-chat-expo/ai-components';

import { type Chat, type ChatState, chatStore } from '../agent/conversation';
import { useAppContext } from '../contexts/AppContext';

const NONE: readonly ToolApprovalRequest[] = [];

const sessionSelector = ({ session }: ChatState) => ({ session });

// What the call would do, in a line, from the arguments the model wrote.
const detailsOf = ({ arguments: written }: ToolApprovalRequest) => {
  try {
    const parsed = JSON.parse(written) as Record<string, unknown>;
    return Object.entries(parsed)
      .filter(([, value]) => typeof value === 'string' || typeof value === 'number')
      .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${String(value)}`)
      .join(' · ');
  } catch {
    return undefined;
  }
};

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

const Requests = ({ session }: { session: Session }) => {
  const approvals = useSyncExternalStore(
    session.conversation.subscribe,
    () => session.conversation.get().approvals ?? NONE,
  );
  if (approvals.length === 0) return null;
  return (
    <View style={styles.container}>
      {approvals.map((request) => (
        <AIToolApprovalCard
          approval={{
            allowTitle: request.approval.allowTitle,
            message: request.approval.message,
            reason: request.reason || detailsOf(request),
            title: request.approval.title,
          }}
          key={request.id}
          onDecide={(allowed) => session.decide(request.id, { allowed })}
        />
      ))}
    </View>
  );
};

/**
 * What the open conversation's session needs the person for, above the composer: why the agent
 * couldn't answer, with a way to ask again, and the calls of the phone's tools that ask first. The
 * phone asks through `responses.create`, so those wait on its session rather than on the reply's
 * step, and the person answers them here.
 */
export const AgentNotices = ({ chat }: { chat: Chat }) => {
  const { session } = useStateStore(chatStore(chat), sessionSelector);
  return session ? (
    <>
      <Failure session={session} />
      <Requests session={session} />
    </>
  ) : null;
};

const styles = StyleSheet.create({
  container: { gap: 8, paddingBottom: 8, paddingHorizontal: 16 },
  failure: { alignItems: 'center', paddingBottom: 8 },
  failureText: { color: '#B00020', fontSize: 14 },
  retryText: { color: '#C8102E', fontSize: 15, fontWeight: '700', paddingTop: 2 },
});
