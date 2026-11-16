import {
  DefaultAttachmentData,
  DefaultChannelData,
  DefaultCommandData,
  DefaultEventData,
  DefaultMemberData,
  DefaultMessageData,
  DefaultPollData,
  DefaultPollOptionData,
  DefaultReactionData,
  DefaultThreadData,
  DefaultUserData,
} from 'stream-chat-expo';
import type { AIUIActionJSON, AIUIDocumentJSON } from 'stream-chat-expo/ai-components';

declare module 'stream-chat' {
  // Module augmentation: the empty interfaces are how the default data types are applied.
  /* eslint-disable @typescript-eslint/no-empty-object-type */

  interface CustomAttachmentData extends DefaultAttachmentData {
    /** A voice memo's length in seconds, which the router writes on it. */
    duration?: number;
  }

  interface CustomChannelData extends DefaultChannelData {}

  interface CustomCommandData extends DefaultCommandData {}

  interface CustomEventData extends DefaultEventData {}

  interface CustomMemberData extends DefaultMemberData {}

  interface CustomUserData extends DefaultUserData {}

  interface CustomMessageData extends DefaultMessageData {
    ai_generated?: boolean;
    /** Set by the router while the agent is still writing a reply. */
    generating?: boolean;
    /** The UI the agent wrote for a reply, which the router moves out of its text. */
    ai_ui?: AIUIDocumentJSON;
    /** What the person did in a reply's UI, on their message that says so. */
    ai_ui_action?: AIUIActionJSON;
    /** The handbook sources a reply drew on, which the backend writes on it. */
    support_message?: {
      sources?: { citation?: string; id: string; title: string; url: string }[];
    };
  }

  interface CustomPollOptionData extends DefaultPollOptionData {}

  interface CustomPollData extends DefaultPollData {}

  interface CustomReactionData extends DefaultReactionData {}

  interface CustomThreadData extends DefaultThreadData {}
}
