# frozen_string_literal: true

module GetStream
  module VisionAgents
    # Conversations to put an agent through, and something that has to be true at the end of
    # each.
    #
    # Server side only: what an agent is tested on is the app's business.
    #
    #   simulation = api.simulations.create(name: "refund", config_id: config["id"],
    #                                       scenario: "Ask for a refund", assertion: "A refund is offered")
    #   run = api.simulations.run(simulation["id"])
    #   run = api.simulations.runs.get(run["id"])
    class Simulations
      # What the simulations have come to.
      attr_reader :runs

      def initialize(client)
        @client = client
        @runs = SimulationRuns.new(client)
      end

      # Writes a simulation down. Nothing is run until #run is called.
      #
      # @param simulation any SimulationRequest field: name, config_id, scenario, assertion,
      #   mode, variations, max_turns, judge_target, caller_target, caller_stt, caller_tts,
      #   caller_voice, tags.
      def create(**simulation)
        @client.post("/v1/agents/simulations", body: simulation)
      end

      def get(id)
        @client.get("/v1/agents/simulations/{id}", path: { id: id })
      end

      def list
        @client.get("/v1/agents/simulations")
      end

      # Replaces a simulation: every field is written, so pass what it now asks rather than
      # what changed. The runs it already has keep their own copy of what they tested.
      def update(id, **simulation)
        @client.put("/v1/agents/simulations/{id}", path: { id: id }, body: simulation)
      end

      # Deletes a simulation. The runs that named it are kept.
      def delete(id)
        @client.delete("/v1/agents/simulations/{id}", path: { id: id })
      end

      # Starts a run. It answers once the run is written rather than once it is over, so read
      # it again with runs.get until its state is no longer running.
      def run(id)
        @client.post("/v1/agents/simulations/{id}/run", path: { id: id })
      end
    end

    # The runs of every simulation, newest first.
    class SimulationRuns
      def initialize(client)
        @client = client
      end

      # One run, with the conversations it had.
      def get(id)
        @client.get("/v1/agents/simulation-runs/{id}", path: { id: id })
      end

      # @param simulation_id [String] only the runs of this simulation; nil, every one's.
      def list(simulation_id: nil, state: nil, limit: nil)
        @client.get("/v1/agents/simulation-runs", query: { simulation_id: simulation_id, state: state, limit: limit })
      end

      # Stops a run in progress, ending the conversations still in flight.
      def cancel(id)
        @client.post("/v1/agents/simulation-runs/{id}/cancel", path: { id: id })
      end
    end

    # What agents remember about the app's users between conversations.
    class Memories
      def initialize(client)
        @client = client
      end

      # Deletes everything remembered about one user: every session's and every agent's,
      # whatever memory filter it was written under. Server side only.
      #
      # @param user_id [String] the user_id of the memory filter the sessions were opened with.
      def truncate(user_id)
        raise ConfigurationError, "truncating memories needs a user id" if user_id.to_s.empty?

        @client.delete("/v1/agents/users/{user_id}/memories", path: { user_id: user_id })
      end
    end
  end
end
