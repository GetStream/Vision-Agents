import React, { useEffect, useState } from 'react';
import { FlatList, Pressable, StyleSheet, Text, View } from 'react-native';

import { SafeAreaView } from 'react-native-safe-area-context';

import type { Schemas } from '@stream-io/stream-agents';
import type { DrawerContentComponentProps } from 'expo-router/drawer';

import { Plus, useStableCallback } from 'stream-chat-expo';

import { agentSessions } from '../agent/agents';
import { useAppContext } from '../contexts/AppContext';

type SessionRow = Schemas['Session'];

const keyExtractor = (session: SessionRow) => session.id;

// When the conversation was last answered, or else started. The router lists conversations by when
// they were last touched, which opening one to read it does too, so a conversation looked at would
// jump to the top. When the agent last answered moves only when it is used.
const lastUsed = (session: SessionRow) =>
  Date.parse(session.last_response_at ?? session.created_at ?? '') || 0;

const byLastUsed = (sessions: readonly SessionRow[]) =>
  [...sessions].sort((a, b) => lastUsed(b) - lastUsed(a));

/**
 * The agent's conversations, most recently used first. Only those kept in writing: a session that
 * was only ever a call has no channel to read back.
 */
export const MenuDrawer = ({ navigation }: DrawerContentComponentProps) => {
  const { conversationsVersion, openConversation, sessionId, startConversation } = useAppContext();
  const [sessions, setSessions] = useState<readonly SessionRow[]>([]);

  useEffect(() => {
    let cancelled = false;
    agentSessions()
      .query({ limit: 50 })
      .then((page) => {
        if (!cancelled) setSessions(byLastUsed(page.items.filter((item) => item.conversation_id)));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [conversationsVersion]);

  const onSelect = useStableCallback((session: SessionRow) => {
    openConversation(session.id, session.conversation_id ?? undefined);
    navigation.closeDrawer();
  });

  const onCreateNewChat = useStableCallback(() => {
    startConversation();
    navigation.closeDrawer();
  });

  const renderItem = ({ item }: { item: SessionRow }) => (
    <Pressable
      onPress={() => onSelect(item)}
      style={({ pressed }) => [styles.row, { opacity: pressed ? 0.6 : 1 }]}
    >
      <Text numberOfLines={1} style={[styles.previewText, item.id === sessionId && styles.current]}>
        {item.title || 'New conversation'}
      </Text>
    </Pressable>
  );

  return (
    <SafeAreaView style={styles.wrapper}>
      <View style={styles.container}>
        <Text style={styles.title}>Conversations</Text>
        <Pressable accessibilityLabel='New conversation' onPress={onCreateNewChat}>
          <Plus size={24} stroke='#000000' />
        </Pressable>
      </View>
      <FlatList data={sessions} keyExtractor={keyExtractor} renderItem={renderItem} />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    borderBottomColor: 'grey',
    borderBottomWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginHorizontal: 12,
    paddingVertical: 16,
  },
  current: { color: '#005FFF' },
  previewText: { fontSize: 15, fontWeight: 'bold' },
  row: { paddingHorizontal: 12, paddingVertical: 8 },
  title: { fontSize: 15, fontWeight: 'bold' },
  wrapper: { backgroundColor: '#FFFFFF', flex: 1 },
});
