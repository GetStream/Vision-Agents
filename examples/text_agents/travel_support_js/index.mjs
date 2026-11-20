// A travel support chat agent, set up from this directory and tested on the dashboard.
//
// instructions.md is who it is, knowledge/travel_policy.md what it knows about bookings, refunds
// and baggage, and skills/itinerary.md how it plans a trip. TableJourney, a public MCP server
// with real restaurants, sights and stays, is a custom connector the app creates once; agent.yaml
// binds it and names the tools the agent may call. This script makes sure the connector exists,
// syncs the agent and opens the dashboard's test page, where you chat with it.
// `lookup_booking` is a function of this process: it stays running and hosts it, so every
// conversation with the agent can call it, the ones the dashboard opens included.
// Needs a router (STREAM_ACCELERATION_URL), Stream credentials and DASHBOARD_BASE_URL, the app's
// agents page on the dashboard.

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { Agent } from "@stream-io/stream-agents";
import { loadFolder } from "@stream-io/stream-agents/node";

const FOLDER = fileURLToPath(new URL(".", import.meta.url));
const CONNECTOR = {
  id: "custom_tablejourney",
  name: "TableJourney",
  description: "Verified restaurants, food festivals and food tours",
  endpoint: "https://tablejourney.com/mcp",
  schemes: ["none"],
};
const CONNECTION_LABEL = "tablejourney";
const BOOKINGS = {
  WF7K2Q: {
    traveller: "Ana Silva",
    trip: "Amsterdam to Lisbon, 12 to 15 June",
    flight: "TP 667, departs 07:05",
    hotel: "Casa do Bairro, 3 nights",
    checked_bags: 1,
    insurance: false,
    status: "confirmed",
  },
};

const agent = new Agent({ folder: await loadFolder(FOLDER) });
await ensureTablejourney(agent.client);
const synced = await agent.sync();

const url = `${process.env.DASHBOARD_BASE_URL.replace(/\/$/, "")}/${synced.config.id}/test?channel=text`;
console.log(`Chat with travel_support: ${url}`);
openInBrowser(url);

agent.client.tools.register({
  name: "lookup_booking",
  description: "Look up a booking by its six-character booking reference, such as WF7K2Q.",
  parameters: { type: "object", properties: { reference: { type: "string" } }, required: ["reference"] },
  run: async ({ reference }) => {
    const key = reference.trim().toUpperCase();
    const booking = BOOKINGS[key];
    return booking ? { reference: key, ...booking } : { error: `no booking with reference ${reference}` };
  },
});

const stop = new AbortController();
process.on("SIGINT", () => stop.abort());
console.log("Hosting lookup_booking; ctrl-c to stop.");
await agent.client.runDispatch(stop.signal);

/** Creates the TableJourney connector and the app's connection to it, once. */
async function ensureTablejourney(client) {
  // Creating it again with the same definition stores nothing new.
  await client.post("/v1/agents/connectors", { body: CONNECTOR });

  const listed = await client.get("/v1/agents/connections", {
    query: { connector_id: CONNECTOR.id, owner_type: "app" },
  });
  if (listed.items.some((connection) => connection.label === CONNECTION_LABEL)) {
    return;
  }

  const connection = await client.post("/v1/agents/connections", {
    body: { connector_id: CONNECTOR.id, owner: { type: "app" }, label: CONNECTION_LABEL },
  });
  // A connector that takes no login is connected by storing empty credentials.
  await client.put("/v1/agents/connections/{id}/credentials", {
    path: { id: connection.id },
    body: { expected_revision: connection.revision },
  });
}

function openInBrowser(url) {
  const command = { darwin: "open", win32: "explorer" }[process.platform] ?? "xdg-open";
  spawn(command, [url], { stdio: "ignore", detached: true }).unref();
}
