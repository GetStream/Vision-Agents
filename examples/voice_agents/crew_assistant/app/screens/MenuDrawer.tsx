import React, { useEffect, useMemo, useState } from 'react';
import { Pressable, SectionList, StyleSheet, Text, View } from 'react-native';

import { SafeAreaView } from 'react-native-safe-area-context';

import type { Schemas } from '@stream-io/stream-agents';
import { router } from 'expo-router';
import { type DrawerContentComponentProps, useDrawerStatus } from 'expo-router/drawer';

import { Plus, useStableCallback } from 'stream-chat-expo';

import { agentSessions } from '../agent/agents';
import { AgentConfig } from '../agent/config';
import { useAppContext } from '../contexts/AppContext';

type SessionRow = Schemas['Session'];

type Section = { data: SessionRow[]; title: string };

const keyExtractor = (session: SessionRow) => session.id;

// When the conversation was last answered, or else started. The router lists conversations by when
// they were last touched, which opening one to read it does too, so a conversation looked at would
// jump to the top. When the agent last answered moves only when it is used.
const lastUsed = (session: SessionRow) =>
  Date.parse(session.last_response_at ?? session.created_at ?? '') || 0;

const DAY = 24 * 60 * 60 * 1000;

// Most recently used first, under Today, Yesterday, Previous 7 days and Older.
const sectionsOf = (sessions: readonly SessionRow[]): Section[] => {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const startOfToday = today.getTime();
  const sections: Section[] = [
    { data: [], title: 'Today' },
    { data: [], title: 'Yesterday' },
    { data: [], title: 'Previous 7 days' },
    { data: [], title: 'Older' },
  ];
  [...sessions]
    .sort((a, b) => lastUsed(b) - lastUsed(a))
    .forEach((session) => {
      const at = lastUsed(session);
      const index =
        at >= startOfToday
          ? 0
          : at >= startOfToday - DAY
            ? 1
            : at >= startOfToday - 7 * DAY
              ? 2
              : 3;
      sections[index].data.push(session);
    });
  return sections.filter((section) => section.data.length > 0);
};

const initialsOf = (name: string) =>
  name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

/**
 * The agent's conversations, most recently used first. Only those kept in writing: a session that
 * was only ever a call has no channel to read back.
 */
export const MenuDrawer = ({ navigation }: DrawerContentComponentProps) => {
  const { conversationsVersion, isReady, openConversation, sessionId, startConversation } =
    useAppContext();
  const [sessions, setSessions] = useState<readonly SessionRow[]>([]);
  const isOpen = useDrawerStatus() === 'open';
  const sections = useMemo(() => sectionsOf(sessions), [sessions]);

  useEffect(() => {
    if (!isReady || !isOpen) return;
    let cancelled = false;
    agentSessions()
      .query({ limit: 50 })
      .then((page) => {
        if (!cancelled) setSessions(page.items.filter((item) => item.conversation_id));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [conversationsVersion, isOpen, isReady]);

  const onSelect = useStableCallback((session: SessionRow) => {
    openConversation(session.id, session.conversation_id ?? undefined);
    router.navigate('/');
    navigation.closeDrawer();
  });

  const onCreateNewChat = useStableCallback(() => {
    startConversation();
    router.navigate('/');
    navigation.closeDrawer();
  });

  const renderItem = ({ item }: { item: SessionRow }) => {
    const isCurrent = item.id === sessionId;
    return (
      <Pressable
        accessibilityRole='button'
        accessibilityState={{ selected: isCurrent }}
        onPress={() => onSelect(item)}
        style={({ pressed }) => [
          styles.row,
          isCurrent ? styles.rowCurrent : pressed ? styles.rowPressed : null,
        ]}
      >
        <Text numberOfLines={1} style={[styles.rowText, isCurrent && styles.rowTextCurrent]}>
          {item.title || 'New conversation'}
        </Text>
      </Pressable>
    );
  };

  const renderSectionHeader = ({ section }: { section: Section }) => (
    <Text accessibilityRole='header' style={styles.sectionTitle}>
      {section.title}
    </Text>
  );

  return (
    <SafeAreaView edges={['top', 'bottom']} style={styles.wrapper}>
      <View style={styles.header}>
        <Text accessibilityRole='header' style={styles.title}>
          Conversations
        </Text>
        <Pressable
          accessibilityRole='button'
          onPress={onCreateNewChat}
          style={({ pressed }) => [styles.newButton, pressed && styles.newButtonPressed]}
        >
          <Plus size={20} stroke={ACCENT} />
          <Text style={styles.newButtonText}>New conversation</Text>
        </Pressable>
      </View>
      <SectionList
        contentContainerStyle={styles.list}
        keyExtractor={keyExtractor}
        renderItem={renderItem}
        renderSectionHeader={renderSectionHeader}
        sections={sections}
        stickySectionHeadersEnabled={false}
      />
      <View style={styles.footer}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{initialsOf(AgentConfig.user.name)}</Text>
        </View>
        <Text numberOfLines={1} style={styles.footerName}>
          {AgentConfig.user.name}
        </Text>
      </View>
    </SafeAreaView>
  );
};

const ACCENT = '#C8102E';
const ACCENT_TINT = '#FDECEE';
const BORDER = '#E6E8EC';
const TEXT = '#0B1220';
const TEXT_SECONDARY = '#5C6370';

const styles = StyleSheet.create({
  avatar: {
    alignItems: 'center',
    backgroundColor: ACCENT_TINT,
    borderRadius: 18,
    height: 36,
    justifyContent: 'center',
    width: 36,
  },
  avatarText: { color: ACCENT, fontSize: 14, fontWeight: '700' },
  footer: {
    alignItems: 'center',
    borderTopColor: BORDER,
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  footerName: { color: TEXT, flex: 1, fontSize: 15, fontWeight: '600' },
  header: { gap: 16, paddingBottom: 8, paddingHorizontal: 20, paddingTop: 16 },
  list: { paddingBottom: 16, paddingHorizontal: 12 },
  newButton: {
    alignItems: 'center',
    alignSelf: 'stretch',
    backgroundColor: ACCENT_TINT,
    borderRadius: 14,
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  newButtonPressed: { opacity: 0.7 },
  newButtonText: { color: ACCENT, fontSize: 15, fontWeight: '600' },
  row: { borderRadius: 12, paddingHorizontal: 12, paddingVertical: 11 },
  rowCurrent: { backgroundColor: ACCENT_TINT },
  rowPressed: { backgroundColor: '#F2F3F5' },
  rowText: { color: TEXT, fontSize: 15 },
  rowTextCurrent: { color: ACCENT, fontWeight: '600' },
  sectionTitle: {
    color: TEXT_SECONDARY,
    fontSize: 13,
    fontWeight: '600',
    paddingBottom: 6,
    paddingHorizontal: 12,
    paddingTop: 18,
  },
  title: { color: TEXT, fontSize: 22, fontWeight: '700' },
  wrapper: { backgroundColor: '#FFFFFF', flex: 1 },
});
