import { Tools } from '@stream-io/stream-agents';

import {
  type AIClientTool,
  type AIClientToolResult,
  AIClientToolRunner,
  type AIToolCallPart,
  completedToolResult,
} from 'stream-chat-expo/ai-components';

/**
 * What only this phone knows: its date, clock, time zone and language. The same tool, by the same
 * name, as the iOS, Android and web assistants, whose instructions call it before a plan with
 * dates.
 */
const localContext: AIClientTool = {
  definition: {
    annotations: { readOnlyHint: true },
    description:
      "The person's current date, local time, timezone and language. Call it before answering anything that depends on today or the time.",
    inputSchema: { properties: {}, type: 'object' },
    name: 'get_local_context',
  },
  run: () => {
    const now = new Date();
    const { locale, timeZone } = Intl.DateTimeFormat().resolvedOptions();
    return completedToolResult(
      { language: locale, local: now.toString(), timeZone, utc: now.toISOString() },
      'Shared the local time',
    );
  },
};

const DEVICE_TOOLS = [localContext];

// What a call says it is doing on the reply's step, by tool.
const DISPLAY_TITLES: Record<string, string> = { get_local_context: "Checking your phone's time" };

/**
 * The results the runner reported, by call, for the session's socket to answer with. A call can
 * reach the socket before its step shows it waiting for this phone, or after.
 */
class Reports {
  private readonly reported = new Map<string, AIClientToolResult>();
  private readonly waiting = new Map<string, (result: AIClientToolResult) => void>();

  report(callId: string, result: AIClientToolResult) {
    const waiting = this.waiting.get(callId);
    this.waiting.delete(callId);
    if (waiting) waiting(result);
    else this.reported.set(callId, result);
  }

  // A call that was cancelled stops being waited on by the session's tools, which leave its
  // result unread.
  wait(callId: string) {
    return new Promise<AIClientToolResult>((resolve) => {
      const reported = this.reported.get(callId);
      this.reported.delete(callId);
      if (reported) return resolve(reported);
      this.waiting.set(callId, resolve);
    });
  }
}

/**
 * The phone's tools, run by the AI components' `AIClientToolRunner` from each reply's steps, as
 * an app whose backend holds the session would run them.
 *
 * Here the phone holds the session too. The router sends it each call over the session's socket
 * and, for a question that named this install, shows the call on the reply as waiting for it. The
 * runner runs it from the reply, and the socket answers the router with what the runner reported.
 * A call no step addresses to this phone, from a question that couldn't name the install, is run
 * by the socket itself.
 */
export const createDeviceTools = ({ clientId, userId }: { clientId: string; userId: string }) => {
  const reports = new Reports();
  const runner = new AIClientToolRunner({ clientId, tools: DEVICE_TOOLS, userId });

  const runUnaddressed = async (name: string, call: AIToolCallPart) => {
    const tool = DEVICE_TOOLS.find(({ definition }) => definition.name === name);
    try {
      return (await tool?.run(call)) ?? { failure: `This phone has no ${name}` };
    } catch {
      return { failure: "The tool couldn't run" };
    }
  };

  const deviceTools = {
    /** Whether the latest question named this install, so its calls wait for this phone. */
    addressed: false,
    runner,
    /** Hands the runner's result to the session's socket. */
    send: (callId: string, result: AIClientToolResult) => {
      reports.report(callId, result);
      return Promise.resolve();
    },
    /** How the router is told of the tools: declared on the session, with the socket's answer. */
    sessionTools: () => {
      const tools = new Tools();
      for (const { description, name, parameters } of runner.registrations) {
        tools.register({
          description,
          displayTitle: DISPLAY_TITLES[name],
          executor: 'client',
          name,
          parameters,
          run: async (input, call) => {
            const { failure, output } = deviceTools.addressed
              ? await reports.wait(call.id)
              : await runUnaddressed(name, {
                  arguments: input,
                  executor: 'client',
                  id: call.id,
                  name,
                  status: 'running',
                });
            if (failure) throw new Error(failure);
            return output;
          },
        });
      }
      return tools;
    },
  };
  return deviceTools;
};
