"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { Button, Empty, Failure, Field, inputStyle } from "@/components/ui";
import { router, type AgentConfig, type KnowledgeDocument } from "@/lib/router";

/**
 * AgentKnowledge is what the business wrote down, which the agent answers out of rather
 * than guessing at.
 *
 * A document is posted once. A url is a subscription, because the page behind it changes
 * and nobody re-posts it, which is why the two are listed differently: the urls can say
 * when they were last read and a document cannot be read back at all.
 */
export function AgentKnowledge({
  config,
  onNamespace,
}: {
  config: AgentConfig;
  onNamespace: (namespace: string) => void;
}) {
  const client = useQueryClient();
  const namespace = config.knowledge_namespace ?? "";

  const urls = useQuery({
    queryKey: ["knowledge-urls", namespace],
    queryFn: () => router.knowledgeUrls(namespace),
    enabled: Boolean(namespace),
  });

  const refresh = () =>
    client.invalidateQueries({ queryKey: ["knowledge-urls", namespace] });

  if (!namespace) {
    return (
      <div className="px-4 py-4">
        <Field
          label="Knowledge namespace"
          hint="What the agent looks things up in. Naming one is what turns the lookup tool on."
        >
          <div className="flex gap-2">
            <input
              className={inputStyle}
              defaultValue={config.name}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  onNamespace(event.currentTarget.value.trim());
                }
              }}
              id="knowledge-namespace"
            />
            <Button
              onClick={() => {
                const field = document.getElementById(
                  "knowledge-namespace",
                ) as HTMLInputElement | null;
                if (field?.value.trim()) {
                  onNamespace(field.value.trim());
                }
              }}
            >
              Use
            </Button>
          </div>
        </Field>
        <p className="mt-3 text-xs text-muted">
          Without one the agent knows only what its instructions say.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 px-4 py-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-muted">
          Looked up in <span className="text-foreground">{namespace}</span>.
        </p>
        <Button variant="quiet" onClick={() => onNamespace("")}>
          Stop looking things up
        </Button>
      </div>

      <Documents namespace={namespace} />

      <div>
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
          Pages it is kept filled from
        </h3>
        <AddUrl namespace={namespace} onAdded={refresh} />

        {urls.isError ? <Failure error={urls.error} /> : null}
        <ul className="mt-2">
          {urls.data?.length === 0 ? (
            <Empty>No pages yet. A url is re-read when you ask, not on a schedule.</Empty>
          ) : null}
          {(urls.data ?? []).map((url) => (
            <li
              key={url.id}
              className="flex items-start gap-4 border-b border-line py-3 first:border-t"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">
                  {url.title || url.url}
                </div>
                <p className="truncate text-xs text-muted">{url.url}</p>
                <p className="mt-1 text-xs text-muted">
                  <State state={url.state} /> · {url.passages} passages
                  {url.last_indexed_at
                    ? ` · read ${new Date(url.last_indexed_at).toLocaleString()}`
                    : " · never read"}
                  {url.error ? ` · ${url.error}` : ""}
                </p>
              </div>
              <div className="flex shrink-0 gap-2">
                <Button
                  variant="quiet"
                  onClick={() =>
                    router.indexKnowledgeUrl(url.id).then(refresh, refresh)
                  }
                >
                  Read again
                </Button>
                <Button
                  variant="danger"
                  onClick={() =>
                    router.deleteKnowledgeUrl(url.id).then(refresh, refresh)
                  }
                >
                  Remove
                </Button>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function State({ state }: { state: string }) {
  const colour =
    state === "indexed"
      ? "text-emerald-600"
      : state === "failed"
        ? "text-red-600"
        : "text-amber-600";
  return <span className={colour}>{state}</span>;
}

/**
 * Documents are posted rather than listed: the passages live in the search index, and
 * there is nothing to read them back out of. So this reports what it just wrote.
 */
function Documents({ namespace }: { namespace: string }) {
  const picker = useRef<HTMLInputElement>(null);

  const ingest = useMutation({
    mutationFn: async (files: File[]) => {
      const documents: KnowledgeDocument[] = await Promise.all(
        files.map(async (file) => ({
          source: file.name,
          text: await file.text(),
        })),
      );
      return router.ingestKnowledge(namespace, documents);
    },
  });

  return (
    <div>
      <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
        Documents
      </h3>
      <p className="mt-1 text-xs text-muted">
        Markdown is cut at its headings, and posting the same filename again replaces what
        it became rather than leaving two versions to be found.
      </p>

      <div className="mt-2 flex items-center gap-3">
        <input
          ref={picker}
          type="file"
          multiple
          accept=".md,.markdown,.txt,text/markdown,text/plain"
          className="hidden"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length) {
              ingest.mutate(files);
            }
            event.target.value = "";
          }}
        />
        <Button
          onClick={() => picker.current?.click()}
          disabled={ingest.isPending}
        >
          {ingest.isPending ? "Reading…" : "Add markdown files"}
        </Button>
        {ingest.data ? (
          <span className="text-xs text-muted">
            {ingest.data.documents} documents became {ingest.data.passages} passages.
          </span>
        ) : null}
      </div>

      {ingest.error ? <Failure error={ingest.error} /> : null}
    </div>
  );
}

function AddUrl({
  namespace,
  onAdded,
}: {
  namespace: string;
  onAdded: () => void;
}) {
  const [url, setUrl] = useState("");

  const add = useMutation({
    mutationFn: () => router.addKnowledgeUrl(namespace, url.trim()),
    onSuccess: () => {
      setUrl("");
      onAdded();
    },
  });

  return (
    <>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          add.mutate();
        }}
      >
        <input
          className={inputStyle}
          type="url"
          placeholder="https://example.com/pricing"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
        />
        <Button type="submit" disabled={add.isPending || !url.trim()}>
          {add.isPending ? "Reading…" : "Add url"}
        </Button>
      </form>
      {add.error ? <Failure error={add.error} /> : null}
    </>
  );
}
