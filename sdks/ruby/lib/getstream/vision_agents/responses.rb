# frozen_string_literal: true

module GetStream
  module VisionAgents
    # The things one or more turns were made of, in the order they happened.
    #
    # Read rather than watched: this is what the backend wrote down, so it reads the same
    # whether the conversation is still going or ended last week. Deltas are not here; a
    # caller who wants to watch words arrive reads Session#events.
    class Items
      include Enumerable

      # How many are read per request, and as many as the router hands over at once.
      PAGE = 200
      CEILING = 1000

      def initialize(client, session_id, response_id = nil)
        @client = client
        @session_id = session_id
        # nil for every turn in the session, set for one turn's own items.
        @response_id = response_id
      end

      # Yields every item, oldest first, fetching a page at a time.
      def each(page: PAGE, &block)
        return enum_for(:each, page: page) unless block

        size = [page, CEILING].min
        offset = 0
        loop do
          read = list(limit: size, offset: offset)
          read.each(&block)
          # A short page is the last page.
          break if read.size < size

          offset += read.size
        end
      end

      # One page, for a caller doing its own paging.
      def list(limit: nil, offset: nil)
        @client.get("/v1/agents/sessions/{id}/responses/items",
                    path: { id: @session_id },
                    query: { response_id: @response_id, limit: limit, offset: offset })
      end

      def all
        to_a
      end
    end

    # One turn, and a way to read what it was made of.
    #
    # Responses#create returns as soon as the agent has started answering rather than when it
    # has finished, so this is a handle on an answer in progress.
    class AgentResponse
      attr_reader :created, :items

      def initialize(client, created)
        @created = created
        @items = Items.new(client, created["session_id"], created["id"])
      end

      # The backend's id for this turn, empty for a session that records nothing. Not the
      # turn_id socket events carry.
      def id
        @created["id"].to_s
      end

      def status
        @created["status"].to_s
      end
    end

    # A session's turns.
    class Responses
      attr_reader :items

      def initialize(client, session_id)
        @client = client
        @session_id = session_id
        @items = Items.new(client, session_id)
      end

      # Asks the agent something and names the turn it answers as.
      #
      # @param images [Array<Hash>] ImageSource hashes, each a url and an optional detail.
      def create(text, images: nil)
        created = @client.post("/v1/agents/sessions/{id}/responses",
                               path: { id: @session_id }, body: { text: text, images: images })
        AgentResponse.new(@client, created)
      end

      # The turns so far, oldest first.
      def list(limit: nil, offset: nil)
        @client.get("/v1/agents/sessions/{id}/responses",
                    path: { id: @session_id }, query: { limit: limit, offset: offset })
      end

      # Goes back to a response and carries on from there.
      #
      # The reply being spoken is abandoned and later turns are no longer listed. The response
      # itself is kept. A conversation kept in Stream Chat cannot be rewound, because the
      # channel would still hold the later turns: fork it at the response instead.
      #
      # @param to [AgentResponse, Hash, String] the response, a response row, any item of it,
      #   or its id.
      def rewind(to)
        response_id = Responses.id_of(to)
        if response_id.empty?
          raise ConfigurationError,
                "that response has no id, which is what a session that records nothing hands back; " \
                "there is nothing to rewind to"
        end

        @client.post("/v1/agents/sessions/{id}/rewind",
                     path: { id: @session_id }, body: { response_id: response_id })
      end

      def self.id_of(response)
        case response
        when AgentResponse then response.id
        when Hash then (response["response_id"] || response["id"]).to_s
        else response.to_s
        end
      end
    end
  end
end
