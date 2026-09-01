"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { Button, Failure, inputStyle } from "@/components/ui";
import {
  router,
  type AgentConfig,
  type Plugin,
  type PluginConnection,
} from "@/lib/router";

/**
 * AgentConnections is the hosted MCP servers this agent may reach.
 *
 * The catalog is the router's. Connecting opens the provider's own login, and the
 * callback lands back on this page with the plugin attached.
 */
export function AgentConnections({ config }: { config: AgentConfig }) {
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const [instanceFor, setInstanceFor] = useState<string | null>(null);
  const [instance, setInstance] = useState("");

  const catalog = useQuery({
    queryKey: ["plugins", query],
    queryFn: () => router.plugins(query),
  });

  const connections = useQuery({
    queryKey: ["plugin-connections", config.id],
    queryFn: () => router.pluginConnections(config.id),
  });

  const byID = useMemo(() => {
    const map = new Map<string, PluginConnection>();
    for (const conn of connections.data ?? []) {
      map.set(conn.plugin_id, conn);
    }
    return map;
  }, [connections.data]);

  const authorize = useMutation({
    mutationFn: ({ plugin, instanceURL }: { plugin: Plugin; instanceURL?: string }) =>
      router.authorizePlugin(config.id, plugin.id, instanceURL),
    onSuccess: (body) => {
      window.location.href = body.authorize_url;
    },
  });

  const disconnect = useMutation({
    mutationFn: (pluginID: string) => router.disconnectPlugin(config.id, pluginID),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["plugin-connections", config.id] });
      client.invalidateQueries({ queryKey: ["config", config.id] });
    },
  });

  const connect = (plugin: Plugin) => {
    if (plugin.instance_required) {
      setInstanceFor(plugin.id);
      setInstance("");
      return;
    }
    authorize.mutate({ plugin });
  };

  const found = useMemo(() => {
    const rows = [...(catalog.data ?? [])];
    rows.sort((a, b) => {
      const aOn = byID.get(a.id)?.status === "connected" ? 0 : 1;
      const bOn = byID.get(b.id)?.status === "connected" ? 0 : 1;
      return aOn - bOn;
    });
    return rows;
  }, [catalog.data, byID]);

  return (
    <div className="px-4 py-4">
      <input
        className={inputStyle}
        type="search"
        placeholder="Search MCP connectors"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />

      {catalog.isError ? <Failure error={catalog.error} /> : null}
      {connections.isError ? <Failure error={connections.error} /> : null}
      {authorize.error ? <Failure error={authorize.error} /> : null}
      {disconnect.error ? <Failure error={disconnect.error} /> : null}

      {found.length === 0 && !catalog.isLoading ? (
        <p className="px-4 py-8 text-center text-sm text-muted">
          Nothing matches {query}.
        </p>
      ) : (
        <ul className="mt-3">
          {found.map((plugin) => {
            const conn = byID.get(plugin.id);
            const asking = instanceFor === plugin.id;
            return (
              <li
                key={plugin.id}
                className="border-b border-line py-2.5 first:border-t"
              >
                <div className="flex items-center gap-4">
                  <div className="min-w-0 flex-1">
                    <span className="text-sm font-medium">{plugin.name}</span>
                    <span className="ml-2 text-sm text-muted">
                      {plugin.description}
                    </span>
                    {conn?.status === "connected" ? (
                      <span className="ml-2 text-xs text-emerald-600">
                        connected
                        {conn.instance_url ? ` · ${conn.instance_url}` : ""}
                      </span>
                    ) : null}
                    {conn?.status === "pending" ? (
                      <span className="ml-2 text-xs text-amber-600">pending</span>
                    ) : null}
                    {conn?.status === "failed" ? (
                      <span className="ml-2 text-xs text-red-600">failed</span>
                    ) : null}
                  </div>
                  <span className="shrink-0 rounded-md border border-line px-1.5 py-0.5 text-xs text-muted">
                    {plugin.category}
                  </span>
                  {conn?.status === "connected" ? (
                    <Button
                      variant="danger"
                      onClick={() => disconnect.mutate(plugin.id)}
                      disabled={disconnect.isPending}
                    >
                      Disconnect
                    </Button>
                  ) : (
                    <Button
                      onClick={() => connect(plugin)}
                      disabled={authorize.isPending}
                    >
                      {authorize.isPending ? "Connecting…" : "Connect"}
                    </Button>
                  )}
                </div>
                {asking ? (
                  <form
                    className="mt-2 flex gap-2"
                    onSubmit={(event) => {
                      event.preventDefault();
                      authorize.mutate({
                        plugin,
                        instanceURL: instance.trim(),
                      });
                    }}
                  >
                    <input
                      className={inputStyle}
                      placeholder={plugin.instance_hint ?? "hostname"}
                      value={instance}
                      onChange={(event) => setInstance(event.target.value)}
                      autoFocus
                    />
                    <Button type="submit" disabled={!instance.trim()}>
                      Continue
                    </Button>
                    <Button
                      variant="quiet"
                      onClick={() => setInstanceFor(null)}
                    >
                      Cancel
                    </Button>
                  </form>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
