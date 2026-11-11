import react from "@vitejs/plugin-react";
import { defineConfig, searchForWorkspaceRoot } from "vite";

export default defineConfig({
  plugins: [react()],
  // One copy of each of these, the page's own, also for packages linked from elsewhere (the SDK
  // and stream-chat-react built from their branches), which bring their own node_modules.
  resolve: {
    dedupe: ["react", "react-dom", "stream-chat", "@stream-io/video-client", "@stream-io/state-store", "use-sync-external-store"],
  },
  server: {
    // The router takes a sign-in's return page only on the origin the session named, and an
    // http one only on localhost, so the page is always served here.
    host: "localhost",
    port: 5173,
    strictPort: true,
    // The trip components are described beside the iOS app, for every platform.
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), "../swift_assistant/app/SwiftAssistant/UIComponents.json"] },
    // swift_assistant's backend answers no browser of another origin, so it is reached
    // through this server instead.
    proxy: {
      "/backend": {
        target: process.env.BACKEND_URL ?? "http://localhost:8098",
        rewrite: (path) => path.replace(/^\/backend/, ""),
      },
    },
  },
  build: { rolldownOptions: { input: { main: "index.html", connected: "connected.html" } } },
});
