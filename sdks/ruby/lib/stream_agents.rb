# frozen_string_literal: true

require_relative "stream_agents/version"
require_relative "stream_agents/errors"
require_relative "stream_agents/generated/spec"
require_relative "stream_agents/backend"
require_relative "stream_agents/socket"
require_relative "stream_agents/client"
require_relative "stream_agents/tool_call"
require_relative "stream_agents/tools"
require_relative "stream_agents/responses"
require_relative "stream_agents/session"
require_relative "stream_agents/folder"
require_relative "stream_agents/inbound"
require_relative "stream_agents/edge"
require_relative "stream_agents/knowledge"
require_relative "stream_agents/agent"
require_relative "stream_agents/dispatch"
require_relative "stream_agents/router"
require_relative "stream_agents/simulations"

# Server-side Ruby SDK for the Stream Agents acceleration backend.
module StreamAgents
end
