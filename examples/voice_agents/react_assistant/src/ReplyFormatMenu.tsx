import { useEffect, useRef, useState } from "react";

import { Assistant } from "./assistant";
import { messageOf } from "./errors";

/** How the agent answers: in Markdown, or with interactive UI in a format. */
export type ReplyFormat = "markdown" | "a2ui" | "openui";

/** The formats this page shows. OpenUI replies show as their text here. */
const SHOWN: { format: ReplyFormat; title: string; description: string }[] = [
  { format: "markdown", title: "Text", description: "Answers in Markdown" },
  { format: "a2ui", title: "Interactive", description: "Answers with cards, maps and forms, in A2UI" },
];

/** Asks swift_assistant's backend how the agent answers. */
export async function replyFormat(): Promise<ReplyFormat> {
  const response = await fetch(`${Assistant.backendUrl}/reply-format`);
  if (!response.ok) throw new Error(`the backend could not say how Juno answers (${response.status})`);
  return ((await response.json()) as { format: ReplyFormat }).format;
}

/** Has the agent answer in `format`, from its next conversation on. */
export async function setReplyFormat(format: ReplyFormat): Promise<void> {
  const response = await fetch(`${Assistant.backendUrl}/reply-format`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format }),
  });
  if (!response.ok) throw new Error(`the backend could not change how Juno answers (${response.status})`);
}

/**
 * How Juno answers, as the iOS app's settings choose it. The choice is the agent's instructions,
 * kept by the backend, so the iOS and Android apps and this page all answer the same way, and a
 * conversation keeps the way it started with.
 */
export function ReplyFormatMenu() {
  const [isOpen, setOpen] = useState(false);
  const [format, setFormat] = useState<ReplyFormat | undefined>(undefined);
  const [failure, setFailure] = useState<string | undefined>(undefined);
  const menu = useRef<HTMLDivElement>(null);

  // A tap anywhere else, or Escape, closes it, as a menu does.
  useEffect(() => {
    if (!isOpen) return;
    const away = (event: PointerEvent) => {
      if (!menu.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    let live = true;
    replyFormat().then(
      (current) => live && setFormat(current),
      (cause: unknown) => live && setFailure(messageOf(cause)),
    );
    return () => {
      live = false;
    };
  }, [isOpen]);

  const choose = async (chosen: ReplyFormat) => {
    const before = format;
    setFormat(chosen);
    setFailure(undefined);
    try {
      await setReplyFormat(chosen);
    } catch (cause) {
      setFormat(before);
      setFailure(messageOf(cause));
    }
  };

  return (
    <div className="reply-format" ref={menu}>
      <button
        aria-expanded={isOpen}
        aria-label="How Juno answers"
        className="icon-button"
        onClick={() => setOpen(!isOpen)}
        type="button"
      >
        ⚙
      </button>
      {isOpen && (
        <div className="reply-format-panel" role="dialog" aria-label="How Juno answers">
          <h3>How Juno answers</h3>
          <div role="radiogroup">
            {SHOWN.map(({ format: each, title, description }) => (
              <label className="reply-format-option" key={each}>
                <input
                  checked={format === each}
                  disabled={format === undefined}
                  name="reply-format"
                  onChange={() => void choose(each)}
                  type="radio"
                />
                <span>
                  <strong>{title}</strong>
                  <span>{description}</span>
                </span>
              </label>
            ))}
          </div>
          {format === "openui" && <p className="reply-format-note">Juno answers in OpenUI, which this page shows as text.</p>}
          <p className="reply-format-note">New conversations answer this way.</p>
          {failure && <p className="failure-pill">{failure}</p>}
        </div>
      )}
    </div>
  );
}
