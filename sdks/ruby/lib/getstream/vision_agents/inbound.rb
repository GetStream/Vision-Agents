# frozen_string_literal: true

require "time"

module GetStream
  module VisionAgents
    # A call that arrived, as the router hands it to a worker.
    #
    # The caller is already in the Stream call; Agent#join with this joins that call rather
    # than creating one, and carries the number they rang so the agent can transfer them.
    class InboundCall
      attr_reader :call_id, :call_type, :called_number, :caller_number, :custom, :at, :session

      def self.from(frame)
        new(call_id: frame["call_id"].to_s, call_type: frame["call_type"].to_s.then { |t| t.empty? ? "default" : t },
            called_number: frame["called_number"].to_s, caller_number: frame["caller_number"].to_s,
            custom: Inbound.strings(frame["custom"]), at: Inbound.time(frame["at"]))
      end

      def initialize(call_id:, call_type: "default", called_number: "", caller_number: "", custom: {}, at: nil)
        @call_id = call_id
        @call_type = call_type
        @called_number = called_number
        @caller_number = caller_number
        @custom = custom
        @at = at
      end

      # Waits until the caller is in the call. Join first: there is nobody to wait for until
      # the agent is in the call they arrived on.
      #
      # @return [Boolean] false if the timeout passed first.
      def wait_for_phone_participant(timeout: nil)
        raise ConfigurationError, "join the call before waiting for the caller" unless @session

        @session.wait_for_participant(timeout: timeout)
      end

      # Called by Agent#join, so the call knows which session to wait on.
      def attach(session)
        @session = session
      end
    end

    # The Python SDK's name for the same thing.
    CallContext = InboundCall

    # A message written to an agent that is not running.
    #
    # One written to an agent that is already running never arrives here: the router answers
    # it from that session, because that agent is the one that knows what has been said.
    InboundMessage = Data.define(:channel_id, :channel_type, :agent_id, :config_id, :text, :message_id,
                                 :user_id, :user_name, :custom, :at) do
      def self.from(frame)
        new(channel_id: frame["channel_id"].to_s,
            channel_type: frame["channel_type"].to_s.then { |t| t.empty? ? "agent" : t },
            agent_id: frame["agent_id"].to_s, config_id: frame["config_id"].to_s, text: frame["text"].to_s,
            message_id: frame["message_id"].to_s, user_id: frame["user_id"].to_s,
            user_name: frame["user_name"].to_s, custom: Inbound.strings(frame["custom"]),
            at: Inbound.time(frame["at"]))
      end

      # The channel as a Stream Chat CID, which is how a session names the conversation.
      def cid
        "#{channel_type}:#{channel_id}"
      end
    end

    # Reading dispatch frames.
    module Inbound
      # The router sends strings, and a value that is not one is dropped rather than rendered,
      # because a number that arrived as JSON should not reach a handler as "17.0".
      def self.strings(value)
        return {} unless value.is_a?(Hash)

        value.select { |_, each| each.is_a?(String) }.transform_keys(&:to_s)
      end

      def self.time(value)
        value.is_a?(String) ? Time.iso8601(value) : nil
      rescue ArgumentError
        nil
      end
    end
  end
end
