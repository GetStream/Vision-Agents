import React, { useCallback, useEffect, useState } from 'react';
import { RefreshControl, ScrollView, StyleSheet, Text, View } from 'react-native';

import { crewBackend } from '../agent/agents';
import { AgentConfig } from '../agent/config';

type InboxItem = {
  id: string;
  kind: 'alert' | 'swap' | 'ticket' | 'check_in';
  from: string;
  title: string;
  details: string;
  status: string;
  at: string;
};

type Pulse = {
  reflections: number;
  average_energy?: number;
  themes: Record<string, number>;
  wins: string[];
};

const LABELS: Record<InboxItem['kind'], string> = {
  alert: '🚨 Call-out',
  check_in: '📅 Check-in',
  swap: '🔁 Shift swap',
  ticket: '🛠️ Repair ticket',
};

/**
 * What the shift leaders and the owner see of what the assistant did: call-out alerts, swap
 * requests, drafted repair tickets and booked check-ins, and the team's pulse from end-of-shift
 * reflections, without names. It signs in as a leader, since the crew backend answers the
 * leaders' inbox only to a user whose crew role is leader.
 */
export default function LeaderView() {
  const [inbox, setInbox] = useState<InboxItem[]>([]);
  const [pulse, setPulse] = useState<Pulse>();
  const [failure, setFailure] = useState<string>();
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const read = (await crewBackend('crew/leader', { userId: AgentConfig.leader.id })) as {
        inbox: InboxItem[];
        pulse: Pulse;
      };
      setInbox(read.inbox);
      setPulse(read.pulse);
      setFailure(undefined);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 3000);
    return () => clearInterval(timer);
  }, [load]);

  const refresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  const themes = Object.entries(pulse?.themes ?? {}).sort(([, a], [, b]) => b - a);

  return (
    <ScrollView
      contentContainerStyle={styles.container}
      refreshControl={<RefreshControl onRefresh={refresh} refreshing={refreshing} />}
    >
      <Text style={styles.meta}>Signed in as {AgentConfig.leader.name}, shift leader</Text>
      {failure ? <Text style={styles.failure}>{failure}</Text> : null}
      <Text style={styles.heading}>Inbox</Text>
      {inbox.length === 0 ? (
        <Text style={styles.empty}>Nothing yet. Call-outs, swaps and tickets show up here.</Text>
      ) : (
        inbox.map((item) => (
          <View key={item.id} style={styles.card} testID={`inbox-${item.kind}`}>
            <Text style={styles.kind}>{LABELS[item.kind] ?? item.kind}</Text>
            <Text style={styles.title}>{item.title}</Text>
            {item.details ? <Text style={styles.details}>{item.details}</Text> : null}
            <Text style={styles.meta}>
              {item.from} · {new Date(item.at).toLocaleTimeString([], { timeStyle: 'short' })} ·{' '}
              {item.status}
            </Text>
          </View>
        ))
      )}

      <Text style={styles.heading}>Team pulse</Text>
      {!pulse || pulse.reflections === 0 ? (
        <Text style={styles.empty}>No end-of-shift reflections yet.</Text>
      ) : (
        <View style={styles.card} testID='team-pulse'>
          <Text style={styles.title}>
            {pulse.reflections} reflection{pulse.reflections === 1 ? '' : 's'} · energy{' '}
            {pulse.average_energy?.toFixed(1)}/5
          </Text>
          <View style={styles.themes}>
            {themes.map(([theme, count]) => (
              <Text key={theme} style={styles.theme}>
                {theme} {count > 1 ? `×${count}` : ''}
              </Text>
            ))}
          </View>
          {pulse.wins.map((win) => (
            <Text key={win} style={styles.details}>
              “{win}”
            </Text>
          ))}
          <Text style={styles.meta}>Shared without names.</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E6E8EC',
    borderRadius: 14,
    borderWidth: 1,
    gap: 4,
    marginBottom: 10,
    padding: 14,
  },
  container: { backgroundColor: '#F7F8FA', flexGrow: 1, padding: 16 },
  details: { color: '#33394A', fontSize: 14 },
  empty: { color: '#5C6370', fontSize: 14, marginBottom: 12 },
  failure: { color: '#C8102E', fontSize: 13, marginTop: 6 },
  heading: { color: '#0B1220', fontSize: 20, fontWeight: '800', marginBottom: 10, marginTop: 8 },
  kind: { color: '#C8102E', fontSize: 12, fontWeight: '700' },
  meta: { color: '#8A919E', fontSize: 12, marginTop: 4 },
  theme: {
    backgroundColor: '#FDECEE',
    borderRadius: 10,
    color: '#C8102E',
    fontSize: 13,
    overflow: 'hidden',
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  themes: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginVertical: 4 },
  title: { color: '#0B1220', fontSize: 15, fontWeight: '700' },
});
