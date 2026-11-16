import '../agent/polyfills';

import React from 'react';

import { StyleSheet } from 'react-native';

import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { KeyboardProvider } from 'react-native-keyboard-controller';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { Drawer } from 'expo-router/drawer';

import type { LocalMessage } from 'stream-chat';
import { Chat, OverlayProvider, useCreateChatClient } from 'stream-chat-expo';

import { StreamTheme } from 'stream-chat-expo/ai-components';

import { streamToken } from '../agent/agents';
import { AgentConfig } from '../agent/config';
import { AppProvider } from '../contexts/AppContext';

import { MenuDrawer } from '../screens/MenuDrawer';

const isMessageAIGenerated = (message: LocalMessage) => !!message.custom?.ai_generated;

export default function Layout() {
  const chatClient = useCreateChatClient({
    apiKey: AgentConfig.streamApiKey,
    tokenOrProvider: streamToken,
    userData: AgentConfig.user,
  });

  if (!chatClient) {
    return null;
  }
  return (
    <SafeAreaProvider>
      <KeyboardProvider>
        <AppProvider client={chatClient}>
          <StreamTheme>
            <GestureHandlerRootView style={styles.container}>
              <OverlayProvider>
                <Chat client={chatClient} isMessageAIGenerated={isMessageAIGenerated}>
                  <Drawer
                    drawerContent={MenuDrawer}
                    screenOptions={{ drawerStyle: { width: 300 } }}
                  >
                    <Drawer.Screen name={'index'} />
                  </Drawer>
                </Chat>
              </OverlayProvider>
            </GestureHandlerRootView>
          </StreamTheme>
        </AppProvider>
      </KeyboardProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
});
