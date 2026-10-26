# frozen_string_literal: true

module GetStream
  module VisionAgents
    # Raised by ToolCall#disallow_interruptions when the call had already been stopped, so the
    # write it guards never starts.
    class CallNotHeldError < Error; end

    # The person whose turn asked for a call.
    ToolUser = Data.define(:id, :name, :custom)
    # The Chat channel a call was made in, and the session's own labels.
    ToolConversation = Data.define(:cid, :custom)

    # The call a tool's block is running for, handed to it as the second argument:
    #
    #   tools.register("book_flight", description: "Book a flight", ...) do |args, tool_call_context|
    #     tool_call_context.update("checking seats")
    #     tool_call_context.disallow_interruptions
    #     bookings.book(args["flight_id"], key: tool_call_context.id) unless tool_call_context.cancelled?
    #   end
    #
    # A Ruby thread cannot be stopped safely from outside, so a call the router cancels or
    # that runs past its timeout keeps running: cancelled? turns true and its answer is
    # dropped. Built without a socket, as in a unit test of the block, its methods do nothing.
    class ToolCall
      # id is the id the model gave the call. A call sent again keeps it, so it is the key to
      # make a side effect happen once.
      attr_reader :id, :session_id, :turn_id, :user, :conversation

      # @param send [#call, nil] sends a frame on the socket that delivered the call.
      def initialize(id: "", session_id: "", turn_id: "", user: ToolUser.new("", "", {}),
                     conversation: ToolConversation.new("", {}), approved: false, send: nil)
        @id = id
        @session_id = session_id
        @turn_id = turn_id
        @user = user
        @conversation = conversation
        @approved = approved
        @send = send
        @lock = Mutex.new
        @answered = ConditionVariable.new
        @state = :running
      end

      # The call a tool_call frame asks for. session_id stands in for a frame that does not
      # name its session, as one from a router from before it said does not.
      def self.from(frame, session_id: "", send: nil)
        user = frame["user"].is_a?(Hash) ? frame["user"] : {}
        conversation = frame["conversation"].is_a?(Hash) ? frame["conversation"] : {}
        new(id: frame["id"].to_s, session_id: (frame["session_id"] || session_id).to_s,
            turn_id: frame["turn_id"].to_s,
            user: ToolUser.new(user["id"].to_s, user["name"].to_s, user["custom"].is_a?(Hash) ? user["custom"] : {}),
            conversation: ToolConversation.new(conversation["cid"].to_s,
                                               conversation["custom"].is_a?(Hash) ? conversation["custom"] : {}),
            approved: frame["approved"] == true, send: send)
      end

      # Whether the caller already said yes to this call out loud, on a voice call.
      def approved?
        @approved
      end

      # Whether the call was stopped: cancelled by the router, out of time, or its socket
      # gone. Its answer will be dropped, so a block can give up early.
      def cancelled?
        @lock.synchronize { @state == :cancelled }
      end

      # Says how the call is going. A voice agent tells the caller in its own words once
      # nobody is talking, and a conversation shows it on the call's step until the call
      # ends. A newer update replaces one not yet said, and the result replaces both.
      def update(text)
        tell(text, false)
      end

      # Has the agent say text as written while the call runs, or post it in a text
      # conversation. A speech-to-speech model is asked for these words and may paraphrase.
      def say(text)
        tell(text, true)
      end

      # Keeps the call going from now on when the caller talks over the agent, which is what
      # a write that must not be left halfway wants. Only the call's timeout and its socket
      # ending stop it after this.
      #
      # @raise [CallNotHeldError] the call had already been stopped.
      def disallow_interruptions
        return nil unless @send && !id.empty?

        @lock.synchronize do
          raise CallNotHeldError, "the call #{id} was stopped before it could be held" if @state == :cancelled
          return nil if @state == :held

          @state = :asking
        end
        @send.call({ type: "tool_hold", tool_call_id: id })
        @lock.synchronize do
          @answered.wait(@lock) while @state == :asking
          raise CallNotHeldError, "the call #{id} was stopped before it could be held" if @state == :cancelled
        end
        nil
      end

      # The router kept the call going. Called by the socket.
      def held!
        @lock.synchronize do
          @state = :held if @state == :asking
          @answered.broadcast
        end
      end

      # The router cancelled the call. A held call keeps going. Called by the socket.
      #
      # @return [Boolean] whether the call is now stopped.
      def cancel!
        @lock.synchronize do
          @state = :cancelled unless @state == :held
          @answered.broadcast
          @state == :cancelled
        end
      end

      # The call ran out of time or lost its socket, which stops even a held call.
      def stop!
        @lock.synchronize do
          @state = :cancelled
          @answered.broadcast
        end
      end

      private

      def tell(text, verbatim)
        return nil unless @send && !id.empty?

        @send.call({ type: "tool_update", tool_call_id: id, text: text, verbatim: verbatim })
        nil
      end
    end
  end
end
