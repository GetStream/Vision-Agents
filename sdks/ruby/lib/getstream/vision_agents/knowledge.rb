# frozen_string_literal: true

module GetStream
  module VisionAgents
    # An agent's knowledge base, as somewhere to put more of it.
    #
    # The namespace is the agent's own name, which is where the knowledge in a synced folder
    # lands, so what is added here is found by the same lookup mid-answer.
    class Knowledge
      # The router queues the read and retries one that fails, so a page can take a while to
      # settle. Past this it is returned still pending rather than waited on forever.
      READ_TIMEOUT = 180
      POLL_INTERVAL = 0.25

      attr_reader :namespace

      def initialize(namespace, client)
        @namespace = namespace
        @client = client
      end

      # Keeps the knowledge base filled from a page published elsewhere.
      #
      # The router queues the read and cuts the page into passages; this waits for it, so what
      # comes back already says whether it worked. It stays a subscription: reading the same
      # url again replaces its passages rather than adding a copy.
      #
      # @return [Hash] the KnowledgeUrl as stored: state, passages, and error if it failed.
      #   Still pending if it was not read within timeout seconds.
      def add_url(url, title: nil, description: nil, timeout: READ_TIMEOUT)
        if @namespace.to_s.empty?
          raise ConfigurationError,
                "a knowledge base is named by the agent it belongs to; build the agent from a config or a folder"
        end

        page = @client.post("/v1/agents/knowledge/urls",
                            body: { namespace: @namespace, url: url, title: title, description: description })
        deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout
        while page["state"] == "pending" && Process.clock_gettime(Process::CLOCK_MONOTONIC) < deadline
          sleep POLL_INTERVAL
          page = @client.get("/v1/agents/knowledge/urls/{id}", path: { id: page.fetch("id") })
        end
        page
      end

      # The pages this knowledge base is kept filled from.
      def urls
        @client.get("/v1/agents/knowledge/urls", query: { namespace: @namespace })
      end
    end
  end
end
