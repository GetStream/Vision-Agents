import { useEffect, useRef } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';

import { useLocalSearchParams, useRouter } from 'expo-router';

import { useAppContext } from '../contexts/AppContext';

/**
 * Where a scanned equipment code leads: a QR code on a machine holds
 * `crewassistant://equipment?asset=FRY-02&code=E-41`, which opens the app here. It starts a new
 * conversation about that machine and its error, and goes back to the chat.
 */
export default function Equipment() {
  const { asset, code } = useLocalSearchParams<{ asset?: string; code?: string }>();
  const { isReady, startConversationWith } = useAppContext();
  const router = useRouter();
  const started = useRef(false);

  useEffect(() => {
    if (!isReady || started.current) return;
    started.current = true;
    const showing = code ? ` It shows error ${code}.` : '';
    startConversationWith(`I scanned the code on ${asset ?? 'a machine'}.${showing}`);
    router.replace('/');
  }, [asset, code, isReady, router, startConversationWith]);

  return (
    <View style={styles.container}>
      <ActivityIndicator />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { alignItems: 'center', flex: 1, justifyContent: 'center' },
});
