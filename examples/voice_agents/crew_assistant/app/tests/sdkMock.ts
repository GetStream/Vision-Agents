import type { AIClientTool, AIClientToolResult } from 'stream-chat-expo/ai-components';

/** The parts of `stream-chat-expo/ai-components` the agent modules call, without its views. */
export const aiComponentsMock = () => ({
  AIClientToolRunner: class {
    readonly registrations: { description: string; name: string; parameters?: unknown }[];
    constructor({ tools }: { tools: AIClientTool[] }) {
      this.registrations = tools.map(({ definition }) => ({
        description: definition.description ?? '',
        name: definition.name,
        parameters: definition.inputSchema,
      }));
    }
  },
  completedToolResult: (output: unknown, summary?: string): AIClientToolResult => ({
    output,
    summary,
  }),
});
