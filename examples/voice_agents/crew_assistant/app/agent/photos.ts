import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import type { AIConversationRequest } from 'stream-chat-expo/ai-components';

// The router takes photos inline, so they are shrunk first.
const MAX_IMAGE_WIDTH = 1280;

/** A photo the person picked, as the data URI the router takes. */
export const toDataUri = async (uri: string) => {
  const { base64 } = await manipulateAsync(uri, [{ resize: { width: MAX_IMAGE_WIDTH } }], {
    base64: true,
    compress: 0.7,
    format: SaveFormat.JPEG,
  });
  return `data:image/jpeg;base64,${base64}`;
};

/**
 * The photos among the files the person picked, as the images the agent's transport sends.
 * Anything else they picked isn't sent: the router takes images only.
 */
export const imagesOf = (attachments: readonly unknown[]) =>
  Promise.all(
    (attachments as AIConversationRequest['attachments'])
      .filter((asset) => !asset.type || asset.type.startsWith('image'))
      .map(async (asset) => ({ url: await toDataUri(asset.uri) })),
  );
