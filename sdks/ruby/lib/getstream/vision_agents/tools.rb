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
    # sent back: a String as it is, anything else as JSON. A block that takes a second
    # argument is handed the ToolCall it runs for.
    class Tools
      Tool = Data.define(:name, :description, :parameters, :executor, :display_title, :interruptible, :timeout,
                         :approval, :run)
      # A {name} in an approval question, filled in from the call's arguments.
      PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)\}/

      def initialize
        @tools = {}
        @lock = Mutex.new
      end

      # @param executor [String] who runs it: "server", the default, or "client" for a tool a
      #   person's device runs. This process still answers it, once the device has reported.
      # @param display_title [String] what a call is doing, in words for the people in the
      #   conversation, such as "Checking your location". At most 80 characters.
      # @param interruptible [Boolean] let the caller talking over the agent cancel a call in
      #   flight. Off, the call keeps running and the model is told it is still going.
      # @param timeout [Numeric] how long one call may take, in seconds, in place of the
      #   session's. A call that takes longer fails, and its answer is dropped.
      # @param approval [String] a question a person answers before each call runs, such as
      #   "Book flight {flight_id}?", where each {name} is an argument filled in from the call.
      #   On a voice call the agent asks it out loud and waits 15 seconds for a yes; in
      #   writing it goes to the app.
      def register(name, description:, parameters: { type: "object", properties: {} }, executor: nil,
                   display_title: nil, interruptible: false, timeout: nil, approval: nil, &run)
        raise ConfigurationError, "a tool needs a name" if name.to_s.empty?
        raise ConfigurationError, "#{name} needs a description, since it is all the model sees" if description.to_s.empty?
        raise ConfigurationError, "#{name} needs a block to run" unless run
        raise ConfigurationError, "#{name} has a timeout that is not a duration" if timeout && !(timeout.is_a?(Numeric) && timeout.positive?)

        approval = check_approval(name, approval&.strip, parameters)
        @lock.synchronize do
          @tools[name.to_s] = Tool.new(name.to_s, description, parameters, executor, display_title, interruptible,
                                       timeout, approval, run)
        end
        self
      end

      def empty?
        @lock.synchronize { @tools.empty? }
      end

      def include?(name)
        @lock.synchronize { @tools.key?(name.to_s) }
      end

      # The tools as a session request declares them.
      def declarations
        @lock.synchronize do
          @tools.values.map do |tool|
            { name: tool.name, description: tool.description, parameters: tool.parameters, executor: tool.executor,
              display_title: tool.display_title, interruptible: (true if tool.interruptible),
              timeout_ms: (([(tool.timeout * 1000).round, 1].max) if tool.timeout),
              approval: ({ title: tool.approval } if tool.approval) }.compact
          end
        end
      end

      # Runs one tool and renders its answer the way tool_result carries it.
      #
      # @raise [ConfigurationError] for a tool nobody registered.
      # @raise [Error] for a call that took longer than the tool's timeout, which is stopped.
      def call(name, arguments, call = ToolCall.new)
        tool = @lock.synchronize { @tools[name] }
        raise ConfigurationError, "no tool is called #{name}" unless tool

        parsed = arguments.to_s.strip.empty? ? {} : JSON.parse(arguments)
        output = tool.timeout ? within(tool, parsed, call) : invoke(tool, parsed, call)
        output.is_a?(String) ? output : JSON.generate(output)
      end

      private

      # A lambda that takes only the arguments is not handed the call it would refuse.
      def invoke(tool, parsed, call)
        tool.run.lambda? && tool.run.arity == 1 ? tool.run.call(parsed) : tool.run.call(parsed, call)
      end

      # Runs the tool on a thread of its own, which is left to finish when it takes too long.
      def within(tool, parsed, call)
        outcome = Thread::Queue.new
        worker = Thread.new do
          outcome << [:output, invoke(tool, parsed, call)]
        rescue StandardError => e
          outcome << [:error, e]
        end
        worker.report_on_exception = false
        kind, value = outcome.pop(timeout: tool.timeout)
        unless kind
          call.stop!
          raise Error, "#{tool.name} took longer than #{tool.timeout}s"
        end
        raise value if kind == :error

        value
      end

      def check_approval(name, approval, parameters)
        return nil if approval.nil?
        raise ConfigurationError, "#{name}'s approval question cannot be blank" if approval.empty?

        properties = (parameters[:properties] || parameters["properties"] || {}).keys.map(&:to_s)
        approval.scan(PLACEHOLDER).flatten.each do |argument|
          next if properties.include?(argument)

          raise ConfigurationError, "#{name}'s approval names {#{argument}}, which is not one of its parameters"
        end
        approval
      end
    end
  end
end
