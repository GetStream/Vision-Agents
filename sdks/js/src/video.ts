/**
 * Stream Video for `session.video()` and `VoiceSession`, opted into by importing this once:
 *
 * ```ts
 * import { VoiceSession } from "@stream-io/calypso/video";
 * ```
 *
 * Its own entry point so `@stream-io/video-client` is resolved only for an app that asked for it.
 */

import { StreamVideoClient } from "@stream-io/video-client";

import { registerVideo } from "./stream.js";

registerVideo(StreamVideoClient);

export { VoiceSession, type VoiceSnapshot, type VoiceStatus } from "./voice-session.js";
