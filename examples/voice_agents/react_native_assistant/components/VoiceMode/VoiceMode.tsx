import { useState } from 'react';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Line } from 'react-native-svg';

import {
  StreamTheme,
  useDictationAudioLevel,
  useTheme,
  type VoiceAgentState,
  VoiceGlow,
  VoiceWave,
} from 'stream-chat-expo/ai-components';

import { useVoiceConversation } from './useVoiceConversation';

const BACKDROP = '#0b0b10';
const COLORS = ['#005fff', '#7b5cff', '#00c2ff'];

const LABELS: Record<VoiceAgentState, string> = {
  connecting: 'Connecting…',
  idle: 'Tap the glow to talk',
  listening: 'Listening',
  speaking: 'Speaking',
  thinking: 'Thinking',
};

const VoiceModeOverlay = ({ onClose }: { onClose: () => void }) => {
  const { top, bottom } = useSafeAreaInsets();
  const { error, listen, phase, userIsTalking } = useVoiceConversation(true);
  const micLevel = useDictationAudioLevel();

  return (
    <StreamTheme style={{ voiceVisualization: { backdropColor: BACKDROP, colors: COLORS } }}>
      <View style={[styles.overlay, { paddingBottom: bottom + 24, paddingTop: top + 12 }]}>
        <View style={styles.header}>
          <Pressable accessibilityLabel='Close voice mode' hitSlop={12} onPress={onClose}>
            <Text style={styles.close}>✕</Text>
          </Pressable>
        </View>
        <Pressable
          accessibilityHint='Starts listening'
          disabled={phase !== 'idle'}
          onPress={listen}
          style={styles.glowContainer}
        >
          <VoiceGlow state={phase} style={styles.glow} />
        </Pressable>
        <Text style={styles.label}>
          {error ? 'Couldn’t hear you. Tap the glow to try again.' : LABELS[phase]}
        </Text>
        <VoiceWave
          accessibilityLabel='Your voice'
          audioLevel={micLevel}
          state={userIsTalking ? 'listening' : 'idle'}
          style={styles.wave}
        />
      </View>
    </StreamTheme>
  );
};

/**
 * A static waveform icon. The button stays still: a live `VoiceGlow` here would redraw every frame
 * for as long as the chat is open.
 */
const VoiceIcon = ({ color, size }: { color: string; size: number }) => (
  <Svg height={size} viewBox='0 0 22 22' width={size}>
    {[
      [3, 8, 14],
      [7, 4, 18],
      [11, 2, 20],
      [15, 5, 17],
      [19, 9, 13],
    ].map(([x, y1, y2]) => (
      <Line
        key={x}
        stroke={color}
        strokeLinecap='round'
        strokeWidth={2.4}
        x1={x}
        x2={x}
        y1={y1}
        y2={y2}
      />
    ))}
  </Svg>
);

/**
 * Opens a full-screen voice conversation with the AI agent: a button after the composer, the size
 * of the attach button before it.
 */
export const VoiceMode = () => {
  const [open, setOpen] = useState(false);
  const {
    theme: { composer },
  } = useTheme();
  const size = composer.inputPillHeight;

  return (
    <>
      <Pressable
        accessibilityLabel='Start voice mode'
        accessibilityRole='button'
        onPress={() => setOpen(true)}
        style={({ pressed }) => [
          styles.button,
          { borderRadius: size / 2, height: size, width: size },
          pressed && styles.buttonPressed,
        ]}
      >
        <VoiceIcon color='#005fff' size={22} />
      </Pressable>
      <Modal animationType='fade' onRequestClose={() => setOpen(false)} visible={open}>
        {open ? <VoiceModeOverlay onClose={() => setOpen(false)} /> : null}
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  button: {
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderColor: '#ececec',
    borderWidth: 1,
    justifyContent: 'center',
    overflow: 'hidden',
  },
  buttonPressed: { opacity: 0.8 },
  close: { color: '#ffffff', fontSize: 22 },
  glow: { aspectRatio: 1, width: '100%' },
  glowContainer: { alignSelf: 'stretch', flex: 1, justifyContent: 'center' },
  header: { alignItems: 'flex-end', paddingHorizontal: 20 },
  label: { color: '#c9cbd3', fontSize: 17, marginVertical: 16, textAlign: 'center' },
  overlay: { backgroundColor: BACKDROP, flex: 1 },
  wave: { alignSelf: 'stretch', height: 140 },
});
