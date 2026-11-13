import React, { useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import {
  AudioRecordingInProgress,
  MessageInputContext,
  type MessageInputContextValue,
  useComponentsContext,
  useStateStore,
} from 'stream-chat-expo';
import {
  AudioRecorderManager,
  type AudioRecorderManagerState,
} from 'stream-chat-react-native-core/src/state-store/audio-recorder-manager';

import { useAppContext } from '../contexts/AppContext';

// One recorder for the app: the button beside the composer starts it, the bar above stops it.
const recorder = new AudioRecorderManager();

// AudioRecordingInProgress reads only the recorder from the composer's context.
const recorderContext = { audioRecorderManager: recorder } as unknown as MessageInputContextValue;

const statusSelector = ({ isStarting, status }: AudioRecorderManagerState) => ({
  isStarting,
  status,
});

const uriOf = ({ recording }: AudioRecorderManagerState) =>
  typeof recording === 'string' ? recording : recording?.getURI();

/** Starts a voice memo, beside the composer. */
export const VoiceMemoButton = () => {
  const { icons } = useComponentsContext();
  const { isStarting, status } = useStateStore(recorder.state, statusSelector);
  if (status === 'recording' || isStarting) return null;
  return (
    <Pressable
      accessibilityLabel='Record a voice memo'
      accessibilityRole='button'
      onPress={() => void recorder.startRecording()}
      style={({ pressed }) => [styles.button, pressed && styles.pressed]}
      testID='voice-memo-start'
    >
      <icons.Mic height={22} stroke='#C8102E' width={22} />
    </Pressable>
  );
};

/**
 * The memo being recorded, above the composer: its time and waveform, from the chat SDK's own
 * recording view, with a way to throw it away and to send it.
 */
export const VoiceMemoBar = () => {
  const { sendVoiceMemo } = useAppContext();
  const { status } = useStateStore(recorder.state, statusSelector);
  const [isSending, setIsSending] = useState(false);
  if (status !== 'recording' && !isSending) return null;

  const cancel = () => void recorder.stopRecording(true);
  const send = async () => {
    setIsSending(true);
    try {
      await recorder.stopRecording();
      const uri = uriOf(recorder.state.getLatestValue());
      recorder.reset();
      if (uri) await sendVoiceMemo(uri);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <View style={styles.bar} testID='voice-memo-bar'>
      {isSending ? (
        <View style={styles.sending}>
          <ActivityIndicator />
          <Text style={styles.sendingText}>Sending your memo</Text>
        </View>
      ) : (
        <>
          <Pressable accessibilityRole='button' onPress={cancel} testID='voice-memo-cancel'>
            <Text style={styles.cancel}>Cancel</Text>
          </Pressable>
          <View style={styles.recording}>
            <MessageInputContext.Provider value={recorderContext}>
              <AudioRecordingInProgress maxDataPointsDrawn={36} />
            </MessageInputContext.Provider>
          </View>
          <Pressable accessibilityRole='button' onPress={send} testID='voice-memo-send'>
            <Text style={styles.send}>Send</Text>
          </Pressable>
        </>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  bar: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E6E8EC',
    borderRadius: 24,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 8,
    marginBottom: 8,
    marginHorizontal: 16,
    paddingHorizontal: 14,
  },
  button: {
    alignItems: 'center',
    backgroundColor: '#FDECEE',
    borderRadius: 22,
    height: 44,
    justifyContent: 'center',
    width: 44,
  },
  cancel: { color: '#5C6370', fontSize: 15, fontWeight: '600' },
  pressed: { opacity: 0.6 },
  recording: { flex: 1, overflow: 'hidden' },
  send: { color: '#C8102E', fontSize: 15, fontWeight: '700' },
  sending: { alignItems: 'center', flexDirection: 'row', gap: 8, paddingVertical: 14 },
  sendingText: { color: '#5C6370', fontSize: 15 },
});
