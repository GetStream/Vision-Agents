/**
 * Stream Video for `session.video()`, opted into by importing this once:
 *
 * ```ts
 * import "@stream-io/vision-agents/video";
 * ```
 *
 * Its own entry point so `@stream-io/video-client` is resolved only for an app that asked for it.
 */

import { StreamVideoClient } from "@stream-io/video-client";

import { registerVideo } from "./stream.js";

registerVideo(StreamVideoClient);
