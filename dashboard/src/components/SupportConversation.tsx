"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ConversationMessage } from "@/components/ConversationMessage";
import { Failure, PageHeading } from "@/components/ui";
import {
  active,
  localRouter,
  mergeMessages,
  type ConversationMessage as Message,
  type ConversationPage,
} from "@/lib/conversation";
import {
  CUSTOMER_ID,
  ROUTER_URL,
  RouterError,
  type Session,
} from "@/lib/router";

export function SupportConversation(props: {
  sessionID: string;
  cid: string;
  agentID: string;
  backend?: string;
  customer?: string;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [connected, setConnected] = useState(false);
  const [ended, setEnded] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [before, setBefore] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const socket = useRef<WebSocket | null>(null);
  const transcript = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const { sessionID, cid, agentID, backend, customer } = props;
  const api = useMemo(() => {
    const base = localRouter(backend, ROUTER_URL);
    const owner = customer ?? CUSTOMER_ID;
    return {
      async request<T>(path: string, body?: unknown): Promise<T> {
        const response = await fetch(base + path, {
          method: body === undefined ? "GET" : "POST",
          headers: {
            "X-Customer-Id": owner,
            ...(body === undefined
              ? {}
              : { "Content-Type": "application/json" }),
          },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: AbortSignal.timeout(20000),
        });
        if (!response.ok)
          throw new RouterError(response.status, await response.text());
        return response.status === 204
          ? (undefined as T)
          : ((await response.json()) as T);
      },
      socket: `${base.replace(/^http/, "ws")}/v1/agents/sessions/${encodeURIComponent(sessionID)}/events?customer_id=${encodeURIComponent(owner)}`,
      history: `/v1/agents/conversations/${encodeURIComponent(cid)}/messages?agent_id=${encodeURIComponent(agentID)}`,
    };
  }, [backend, customer, sessionID, cid, agentID]);

  useEffect(() => {
    let disposed = false;
    let reconnect: ReturnType<typeof setTimeout>;
    let attempt = 0;
    // Frames received while the initial snapshot loads take precedence over that snapshot.
    let buffered: Message[] | null = [];
    const history = async () => {
      const page = await api.request<ConversationPage>(api.history);
      if (disposed) return;
      setMessages((current) =>
        mergeMessages(mergeMessages(current, page.messages), buffered ?? []),
      );
      buffered = null;
      setBefore(page.before);
      setLoading(false);
    };
    const connect = async () => {
      try {
        const session = await api.request<Session>(
          `/v1/agents/sessions/${encodeURIComponent(sessionID)}`,
        );
        if (disposed) return;
        if (session.conversation_id !== cid || session.agent_id !== agentID)
          throw new Error("Session does not belong to this conversation.");
        if (session.state === "ended") {
          setEnded(true);
          await history();
          return;
        }
        const connection = new WebSocket(api.socket);
        socket.current = connection;
        connection.onopen = () => {
          if (disposed) {
            connection.close();
            return;
          }
          attempt = 0;
          setConnected(true);
          setError(null);
          buffered = [];
          history().catch((err) => {
            if (!disposed) setError(err);
          });
        };
        connection.onmessage = (event) => {
          const frame = JSON.parse(event.data);
          if (
            frame.type !== "conversation_updated" ||
            frame.conversation_id !== cid
          )
            return;
          const message = frame.message as Message;
          buffered?.push(message);
          setMessages((current) => mergeMessages(current, [message]));
        };
        connection.onclose = () => {
          if (disposed) return;
          setConnected(false);
          reconnect = setTimeout(
            connect,
            Math.min(1000 * 2 ** attempt++, 15000),
          );
        };
      } catch (err) {
        if (disposed) return;
        if (err instanceof RouterError && err.status === 404) {
          setEnded(true);
          history().catch((err) => {
            if (!disposed) setError(err);
          });
        } else {
          setError(err);
          reconnect = setTimeout(
            connect,
            Math.min(1000 * 2 ** attempt++, 15000),
          );
        }
      }
    };
    // History remains readable even while the owning TUI/backend is reconnecting.
    history()
      .catch((err) => {
        if (!disposed) {
          setError(err);
          setLoading(false);
        }
      })
      .then(() => {
        if (!disposed) connect();
      });
    return () => {
      disposed = true;
      clearTimeout(reconnect);
      socket.current?.close();
      socket.current = null;
    };
  }, [api, sessionID, cid, agentID]);

  useEffect(() => {
    if (following.current && transcript.current)
      transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [messages]);

  const busy =
    sending ||
    messages.some((message) => active(message.state) && !message.finished_at);
  const submit = async () => {
    if (!text.trim() || busy || !connected) return;
    setSending(true);
    setError(null);
    try {
      await api.request(
        `/v1/agents/sessions/${encodeURIComponent(sessionID)}/respond`,
        { text },
      );
      setText("");
      following.current = true;
    } catch (err) {
      setError(err);
    } finally {
      setSending(false);
    }
  };
  const loadOlder = async () => {
    if (!before || loadingOlder) return;
    setLoadingOlder(true);
    try {
      const page = await api.request<ConversationPage>(
        `${api.history}&before=${encodeURIComponent(before)}`,
      );
      following.current = false;
      const height = transcript.current?.scrollHeight ?? 0;
      setMessages((current) => mergeMessages(page.messages, current));
      setBefore(page.before);
      requestAnimationFrame(() => {
        if (transcript.current)
          transcript.current.scrollTop +=
            transcript.current.scrollHeight - height;
      });
    } catch (err) {
      setError(err);
    } finally {
      setLoadingOlder(false);
    }
  };

  return (
    <>
      <PageHeading
        title="Support conversation"
        description="Questions, answers, and agent activity · text only"
        action={
          <span className="rounded-full border border-line px-3 py-1 text-xs">
            {connected
              ? "● Live"
              : ended
                ? "Saved conversation"
                : "Reconnecting…"}
          </span>
        }
      />
      <div className="mb-5 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted">
        <span>
          Channel <code className="break-all text-foreground">{cid}</code>
        </span>
        <span>
          Agent <code>{agentID}</code>
        </span>
      </div>
      {error ? (
        <div className="mb-4">
          <Failure error={error} />
        </div>
      ) : null}
      <section className="overflow-hidden rounded-2xl border border-line bg-background">
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <h2 className="text-sm font-semibold">Conversation</h2>
          <span className="text-xs text-muted">
            History stored in Stream Chat
          </span>
        </div>
        <div
          ref={transcript}
          onScroll={() => {
            const view = transcript.current;
            if (view)
              following.current =
                view.scrollHeight - view.scrollTop - view.clientHeight < 80;
          }}
          className="h-[min(65vh,48rem)] min-h-64 space-y-5 overflow-y-auto p-5"
          aria-label="Conversation transcript"
        >
          {before && (
            <button
              className="w-full text-sm text-sky-600"
              disabled={loadingOlder}
              onClick={loadOlder}
            >
              {loadingOlder ? "Loading…" : "Load earlier messages"}
            </button>
          )}
          {messages.length === 0 && (
            <p className="py-16 text-center text-sm text-muted">
              {loading
                ? "Loading conversation…"
                : "Ask a question in the TUI or here to get started."}
            </p>
          )}
          {messages.map((message) => (
            <ConversationMessage
              key={message.id}
              message={message}
              live={connected}
            />
          ))}
        </div>
        <form
          className="border-t border-line bg-surface p-4"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
          <label className="sr-only" htmlFor="support-question">
            Your question
          </label>
          <textarea
            id="support-question"
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={6000}
            rows={3}
            disabled={!connected}
            placeholder={
              ended
                ? "Resume this channel in ./support to continue"
                : "Ask about the SDK…"
            }
            className="w-full resize-y rounded-xl border border-line bg-background px-3 py-2 text-sm outline-sky-500 disabled:opacity-50"
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="text-xs text-muted">
              {ended
                ? "Reopen with ./support --channel " + cid + " --dashboard"
                : "Enter to send · Shift-Enter for a new line · Keep the TUI open for local tools"}
            </p>
            {busy && connected ? (
              <button
                type="button"
                onClick={() =>
                  socket.current?.send(JSON.stringify({ type: "interrupt" }))
                }
                className="rounded-lg border border-line px-4 py-2 text-sm"
              >
                Cancel response
              </button>
            ) : (
              <button
                disabled={!connected || busy || !text.trim()}
                className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
              >
                Send
              </button>
            )}
          </div>
        </form>
      </section>
    </>
  );
}
