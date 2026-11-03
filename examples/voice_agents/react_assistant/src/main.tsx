import { createRoot } from "react-dom/client";
import "stream-chat-react/css/index.css";
import "@openuidev/react-ui/components.css";
import "stream-chat-react/css/ai-components.css";

import { App } from "./App";
import "./styles.css";

// No StrictMode: it runs every effect twice in development, and signing in connects Stream Chat
// as the user once.
createRoot(document.getElementById("root")!).render(<App />);
