# frozen_string_literal: true

require_relative "vision_agents/version"
require_relative "vision_agents/errors"
require_relative "vision_agents/generated/spec"
require_relative "vision_agents/backend"
require_relative "vision_agents/socket"
require_relative "vision_agents/client"
require_relative "vision_agents/tools"
require_relative "vision_agents/responses"
require_relative "vision_agents/session"
require_relative "vision_agents/folder"
require_relative "vision_agents/inbound"
require_relative "vision_agents/edge"
require_relative "vision_agents/knowledge"
require_relative "vision_agents/agent"
require_relative "vision_agents/dispatch"
require_relative "vision_agents/router"

module GetStream
  # Server-side Ruby SDK for the Vision Agents acceleration backend.
  module VisionAgents
  end
end
