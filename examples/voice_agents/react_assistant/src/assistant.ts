import { Client, type Sessions } from "@stream-io/calypso";
import "@stream-io/calypso/chat";
import "@stream-io/calypso/video";
import { StreamChat } from "stream-chat";

/**
 * Where the router is, where the token backend is, and who this page is.
 *
 * There is no sign-in. The router runs in the mode where it trusts the customer and user ids it
 * is given, `ROUTER_AUTH_MODE=proxy`, so these are the whole of the configuration. In front of a
 * real deployment the router is reached by the key and the same token, and nothing else here
 * would change.
 */
export const Assistant = {
  /** The router. */
  routerUrl: import.meta.env.VITE_ROUTER_URL || "http://localhost:8080",

  /**
   * This app's backend, swift_assistant's `go run ./backend`, through the dev server. A page holds
   * no secret to sign a Stream token with and may not reach its person's Gmail connection on the
   * router, so it asks this.
   */
  backendUrl: "/backend",

  /**
   * Where the sign-ins a conversation asks for go back to, kept with its session. The router hands
   * the sign-in over only to a page of this origin.
   */
  returnUrl: `${window.location.origin}/connected.html`,

  /** The customer swift_assistant's `go run ./configure` stored the agent under. */
  customerId: "examples",

  /** The `STREAM_API_KEY` the router runs with, which the conversations' channels and calls are in. */
  streamApiKey: import.meta.env.VITE_STREAM_API_KEY ?? "",

  /** The agent swift_assistant's `go run ./configure` stored, by the name in its `agent.yaml`. */
  agentName: "swift_assistant",

  /**
   * The model a written conversation runs on. Grok reads photos and streams its reasoning, which
   * is what fills a reply's thinking panel. A call keeps the agent's faster model.
   */
  chatModel: "xai/grok-4.7",

  /**
   * Who this page is. By default the same person the iOS and Android apps are, so all three list
   * the same conversations; `VITE_USER_ID` and `VITE_USER_NAME` make it somebody else, with
   * conversations and connected accounts of their own.
   */
  user: {
    id: import.meta.env.VITE_USER_ID || "juno-owner",
    name: import.meta.env.VITE_USER_NAME || "Alex",
  },
} as const;

/** The router, as this page's user. */
export const agents = new Client({
  url: Assistant.routerUrl,
  customerId: Assistant.customerId,
  apiKey: Assistant.streamApiKey,
});

/**
 * Stream Chat, connected as the same user, which Stream's components and every conversation's
 * channel use. Handed to the SDK, so it opens no second one.
 */
export const chatClient = new StreamChat(Assistant.streamApiKey || "missing-key");

/** Says who this page is to the router and to Stream Chat. Resolves once both are ready. */
export async function signIn(): Promise<void> {
  await agents.setUser(Assistant.user, streamToken);
  await chatClient.connectUser(Assistant.user, streamToken);
  agents.use({ chat: chatClient });
}

/** The agent's conversations: opening one, and listing and searching the ones there were. */
export function sessions(): Sessions {
  return agents.agent(Assistant.agentName).sessions;
}

/** Asks the backend for a Stream user token, which opens the chat and joins the call. */
async function streamToken(): Promise<string> {
  const response = await fetch(`${Assistant.backendUrl}/stream-token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: Assistant.user.id }),
  });
  if (!response.ok) {
    throw new Error("the backend would not mint a Stream token: is `go run ./backend` running?");
  }
  const minted = (await response.json()) as { token?: string };
  return minted.token ?? "";
}
