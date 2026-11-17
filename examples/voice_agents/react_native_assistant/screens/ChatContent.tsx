import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, { FadeIn, FadeOut } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useStateStore } from 'stream-chat-expo';

import { createA2UIRenderer } from 'stream-chat-expo/a2ui';
import {
  type AIConversationController,
  type AIConversationControllerState,
  AISuggestions,
  AITypingIndicatorView,
  aiUIActionJSON,
  AIUIHost,
  ComposerView,
  isAIConversationAnswering,
  useAIConversationState,
} from 'stream-chat-expo/ai-components';
import { AIConversation } from 'stream-chat-expo/ai-components/conversation';

import { type Chat, type ChatState, chatStore, keyOf } from '../agent/conversation';
import { chatOptions, STARTERS } from '../chatOptions';
import { tripComponents } from '../components/TripComponents';
import { VoiceMode } from '../components/VoiceMode';
import { useAppContext } from '../contexts/AppContext';

const errorSelector = ({ error }: AIConversationControllerState) => ({ error });

/**
 * The composer, with what rides the keyboard above it: why something failed, with a way to try
 * again, and the starters, which share the composer's draft and picked option. And the way into
 * voice mode beside it.
 */
const ComposerArea = ({
  chat,
  conversation,
  failure,
}: {
  chat: Chat;
  conversation: AIConversationController;
  failure: string | undefined;
}) => {
  const { error } = useAIConversationState(conversation, errorSelector);

  const safeAreaInsets = useSafeAreaInsets();
  const insets = useMemo(
    () => ({
      ...safeAreaInsets,
      bottom:
        safeAreaInsets.bottom +
        (Platform.OS === 'android' ? (StatusBar.currentHeight ?? 0) * 2 : 0),
    }),
    [safeAreaInsets],
  );

  return (
    <ComposerView
      bottomSheetInsets={insets}
      chatOptions={chatOptions}
      conversation={conversation}
      header={
        <>
          {failure ? <Text style={styles.failureText}>{failure}</Text> : null}
          {error ? (
            <Pressable accessibilityRole='button' onPress={() => conversation.retry()}>
              <Text style={styles.failureText}>{error.message}</Text>
              <Text style={styles.retryText}>Try again</Text>
            </Pressable>
          ) : null}
          <Starters chat={chat} conversation={conversation} />
        </>
      }
      trailing={<VoiceMode />}
    />
  );
};

const EmptyState = () => <Text style={styles.emptyText}>What can I help with ?</Text>;

// A conversation still loading isn't empty: it shows no starters meanwhile.
const isEmptySelector = ({ isLoaded, messages }: ChatState) => ({
  isEmpty: isLoaded && messages.length === 0,
});

const pendingSelector = ({ pendingMessage }: AIConversationControllerState) => ({
  pendingMessage,
});

/**
 * Starters, above the composer, until the conversation has a first question. One is asked with the
 * option picked in the composer, as the composer would ask it.
 */
const Starters = ({
  chat,
  conversation,
}: {
  chat: Chat;
  conversation: AIConversationController;
}) => {
  const { isEmpty } = useStateStore(chatStore(chat), isEmptySelector);
  const { pendingMessage } = useAIConversationState(conversation, pendingSelector);
  if (!isEmpty || pendingMessage) return null;
  const onSelect = (text: string) => {
    const { chatOption } = conversation.composer.getLatestValue();
    conversation.send({ chatOption: chatOption ?? undefined, text });
  };
  return <AISuggestions onSelect={onSelect} suggestions={STARTERS} />;
};

const loadedSelector = ({ isLoaded }: ChatState) => ({ isLoaded });

/** The open conversation, as its controller has it. */
const Conversation = ({
  chat,
  conversation,
}: {
  chat: Chat;
  conversation: AIConversationController;
}) => {
  const { agentState } = useAppContext();
  const { isLoaded } = useStateStore(chatStore(chat), loadedSelector);

  // The UI the agent writes for its replies, in A2UI, with the trip planner's own components. One
  // host per conversation, for as long as it shows, so what a reply's UI holds survives scrolling.
  // What the person does in it goes to the agent as their next message.
  const [ui] = useState(
    () =>
      new AIUIHost({
        components: tripComponents,
        renderers: [createA2UIRenderer()],
        send: (action) => {
          if (isAIConversationAnswering(conversation.state.getLatestValue().status)) {
            throw new Error('The agent is still answering');
          }
          const custom = { uiAction: aiUIActionJSON(action) };
          conversation.send({ custom, text: action.label });
        },
      }),
  );

  // What the agent is doing while it runs skills, under the latest turn.
  const footer =
    agentState.kind === 'working' ? (
      <AITypingIndicatorView
        text={`Working on ${agentState.skills.map((skill) => skill.replace(/_/g, ' ')).join(' and ')}`}
      />
    ) : undefined;

  return (
    <AIConversation
      conversation={conversation}
      emptyContent={isLoaded ? <EmptyState /> : <ActivityIndicator />}
      footer={footer}
      ui={ui}
    />
  );
};

export const ChatContent = () => {
  const { chat, conversation, failure } = useAppContext();
  const { bottom } = useSafeAreaInsets();

  if (!chat || !conversation) {
    return (
      <View style={styles.emptyContainer}>
        {failure ? <Text style={styles.failureText}>{failure}</Text> : <ActivityIndicator />}
      </View>
    );
  }

  return (
    <Animated.View
      key={keyOf(chat)}
      style={[styles.wrapper, { paddingBottom: bottom }]}
      entering={FadeIn.duration(200)}
      exiting={FadeOut.duration(200)}
    >
      {/* The keyboard is the SDK's: the conversation makes room for it, the composer rides it. */}
      <Conversation chat={chat} conversation={conversation} />
      <ComposerArea chat={chat} conversation={conversation} failure={failure} />
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  wrapper: { flex: 1, backgroundColor: '#fcfcfc' },
  emptyContainer: {
    flex: 1,
    width: '100%',
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: { fontSize: 24, fontWeight: 'bold' },
  failureText: { color: '#B00020', fontSize: 15, paddingHorizontal: 24, textAlign: 'center' },
  retryText: {
    color: '#005FFF',
    fontSize: 15,
    fontWeight: '600',
    paddingTop: 4,
    textAlign: 'center',
  },
});
