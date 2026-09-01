import { Target } from "@/components/AgentModelFields";
import { Field, inputStyle } from "@/components/ui";
import { type AgentConfig, type SimulationRequest } from "@/lib/router";

/** A simulation nobody has filled in yet. */
export const blank: SimulationRequest = {
  name: "",
  mode: "text",
  config_id: "",
  scenario: "",
  assertion: "",
  variations: 1,
};

/**
 * SimulationForm is what to ask, who to ask it of and what has to be true afterwards.
 *
 * The scenario is a brief rather than a script, which is the thing worth saying on the page
 * itself: somebody who writes it as three literal sentences to read out will be surprised
 * by what the caller does with it.
 */
export function SimulationForm({
  simulation,
  configs,
  onChange,
}: {
  simulation: SimulationRequest;
  configs: AgentConfig[];
  onChange: (simulation: SimulationRequest) => void;
}) {
  const set = (changed: Partial<SimulationRequest>) =>
    onChange({ ...simulation, ...changed });

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Name">
          <input
            className={inputStyle}
            placeholder="Changes their mind mid-order"
            value={simulation.name}
            onChange={(event) => set({ name: event.target.value })}
          />
        </Field>

        <Field
          label="Type"
          hint="Text tests everything between hearing and answering. Audio generates speech and runs the whole pipeline."
        >
          <select
            className={inputStyle}
            value={simulation.mode ?? "text"}
            onChange={(event) =>
              set({ mode: event.target.value as SimulationRequest["mode"] })
            }
          >
            <option value="text">Text</option>
            <option value="audio">Audio</option>
          </select>
        </Field>
      </div>

      <Field label="Agent" hint="The agent this is run against.">
        <select
          className={inputStyle}
          value={simulation.config_id}
          onChange={(event) => set({ config_id: event.target.value })}
        >
          <option value="">Pick an agent</option>
          {configs.map((config) => (
            <option key={config.id} value={config.id}>
              {config.name}
            </option>
          ))}
        </select>
      </Field>

      <Field
        label="What to ask"
        hint="In your own words, over as many turns as it takes. This is a brief for the caller rather than a script, so it can describe things that depend on what the agent says back."
      >
        <textarea
          className={`${inputStyle} h-28 resize-y`}
          placeholder="In 3 steps: place an order for pasta bolognese, then once the order is handled change your mind to a pepperoni pizza, then tell them to deliver at 8pm."
          value={simulation.scenario}
          onChange={(event) => set({ scenario: event.target.value })}
        />
      </Field>

      <Field
        label="Evaluation"
        hint="One question. The judge answers only this, so an agent that was rude but did the thing still passes."
      >
        <textarea
          className={`${inputStyle} h-20 resize-y`}
          placeholder="Was an order placed for a pepperoni pizza with delivery at 8pm?"
          value={simulation.assertion}
          onChange={(event) => set({ assertion: event.target.value })}
        />
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Variations"
          hint="Expanding has a model write other ways of asking the same thing. Every fact is kept; only the wording changes."
        >
          <select
            className={inputStyle}
            value={simulation.variations === 10 ? "10" : "1"}
            onChange={(event) => set({ variations: Number(event.target.value) })}
          >
            <option value="1">None</option>
            <option value="10">Expand 10x</option>
          </select>
        </Field>

        <Field label="Turns" hint="How many times the caller may speak before it gives up.">
          <input
            className={inputStyle}
            type="number"
            min={1}
            max={30}
            value={simulation.max_turns ?? 12}
            onChange={(event) => set({ max_turns: Number(event.target.value) })}
          />
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Target
          modality="llm"
          label="Judge"
          hint="The model that rules on the conversations. Empty takes a quality tier, since nobody is waiting for it."
          placeholder="multilingual-high-accuracy"
          value={simulation.judge_target}
          onChange={(judge_target) => set({ judge_target })}
        />
        <Target
          modality="llm"
          label="Caller"
          hint="The model that plays the person on the phone. Empty takes a fast tier."
          placeholder="llm-fast"
          value={simulation.caller_target}
          onChange={(caller_target) => set({ caller_target })}
        />
      </div>
    </div>
  );
}
