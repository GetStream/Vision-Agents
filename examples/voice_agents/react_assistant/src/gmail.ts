import { Assistant } from "./assistant";

/**
 * Sending from the person's Gmail.
 *
 * The person connects Gmail in the chat, the first time the agent reaches for it: its call waits
 * on a sign-in shown on its step, and the router keeps the Google credential, which the page
 * never sees. Google's Gmail MCP server writes drafts but cannot send, so a send goes through the
 * app's backend, which posts to Gmail's REST API through the router's proxy for the same
 * connection.
 */
export async function sendEmail(to: string, subject: string, body: string): Promise<string> {
  const response = await fetch(`${Assistant.backendUrl}/gmail/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: Assistant.user.id, to, subject, body }),
  });
  if (!response.ok) {
    const said = (await response.text()).trim();
    throw new Error(said || "The backend could not reach Gmail.");
  }
  const sent = (await response.json()) as { id?: string };
  return sent.id ?? "";
}
