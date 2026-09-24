# frozen_string_literal: true

require "json"

module GetStream
  module VisionAgents
    # The caller's own functions, which the model is offered and this process runs.
    #
    #   agent.tools.register("get_weather", description: "Weather for a city",
    #                        parameters: { type: "object", properties: { city: { type: "string" } } }) do |args|
    #     weather_in(args["city"])
    #   end
    #
    # The model asks for one over the session socket and waits. What the block returns is
    # sent back: a String as it is, anything else as JSON.
    class Tools
      Tool = Data.define(:name, :description, :parameters, :run)

      def initialize
        @tools = {}
        @lock = Mutex.new
      end

      def register(name, description:, parameters: { type: "object", properties: {} }, &run)
        raise ConfigurationError, "a tool needs a name" if name.to_s.empty?
        raise ConfigurationError, "#{name} needs a description, since it is all the model sees" if description.to_s.empty?
        raise ConfigurationError, "#{name} needs a block to run" unless run

        @lock.synchronize { @tools[name.to_s] = Tool.new(name.to_s, description, parameters, run) }
        self
      end

      def empty?
        @lock.synchronize { @tools.empty? }
      end

      # The tools as a session request declares them.
      def declarations
        @lock.synchronize do
          @tools.values.map { |tool| { name: tool.name, description: tool.description, parameters: tool.parameters } }
        end
      end

      # Runs one tool and renders its answer the way tool_result carries it.
      #
      # @raise [ConfigurationError] for a tool nobody registered.
      def call(name, arguments)
        tool = @lock.synchronize { @tools[name] }
        raise ConfigurationError, "no tool is called #{name}" unless tool

        parsed = arguments.to_s.strip.empty? ? {} : JSON.parse(arguments)
        output = tool.run.call(parsed)
        output.is_a?(String) ? output : JSON.generate(output)
      end
    end
  end
end
