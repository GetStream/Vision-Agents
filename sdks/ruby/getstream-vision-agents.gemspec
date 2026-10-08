# frozen_string_literal: true

require_relative "lib/getstream/vision_agents/version"

Gem::Specification.new do |spec|
  spec.name = "getstream-vision-agents"
  spec.version = GetStream::VisionAgents::VERSION
  spec.authors = ["Stream"]
  spec.email = ["support@getstream.io"]
  spec.summary = "Server-side Ruby SDK for Stream's Vision Agents acceleration backend"
  spec.description = "Configure agents, open sessions, answer inbound calls and messages, " \
                     "place outbound calls and sync agent folders against the Vision Agents backend."
  spec.homepage = "https://github.com/GetStream/Vision-Agents"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.3"

  spec.metadata = {
    "homepage_uri" => spec.homepage,
    "source_code_uri" => "https://github.com/GetStream/Vision-Agents/tree/main/sdks/ruby",
    "bug_tracker_uri" => "https://github.com/GetStream/Vision-Agents/issues",
    "rubygems_mfa_required" => "true"
  }

  spec.files = Dir["lib/**/*.rb", "README.md"]
  spec.require_paths = ["lib"]

  spec.add_dependency "getstream-ruby", "~> 12.1"
  spec.add_dependency "websocket-driver", "~> 0.8"
end
