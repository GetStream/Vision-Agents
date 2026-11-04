import react from "@vitejs/plugin-react";
import { defineConfig, searchForWorkspaceRoot } from "vite";

export default defineConfig({
  plugins: [react()],
  // The SDK comes from this checkout, with its own node_modules beside it: one copy of each
  // Stream client and of React, or chat and video would each be loaded twice.
  resolve: { dedupe: ["react", "react-dom", "stream-chat", "@stream-io/video-client"] },
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
  build: { rollupOptions: { input: { main: "index.html", connected: "connected.html" } } },
});
