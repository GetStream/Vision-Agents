# frozen_string_literal: true

require "digest"
require "json"
require "psych"
require "time"

module GetStream
  module VisionAgents
    # A kind of work worth handing to the slower model.
    #
    # There is nothing behind a skill but a better model and more time. What it declares is
    # the description the fast model chooses by, and the instructions the slow one answers
    # under.
    #
    # @!attribute deadline
    #   How long the work may run before it is abandoned, in seconds. nil leaves the
    #   backend's default.
    Skill = Data.define(:name, :description, :instructions, :capture_video, :deadline) do
      def initialize(name:, description:, instructions:, capture_video: false, deadline: nil)
        super
      end

      # The skill as a session request or a sync request takes it.
      def request
        rendered = { name: name, description: description, instructions: instructions,
                     capture_video: capture_video }
        rendered[:deadline_ms] = (deadline * 1000).to_i if deadline&.positive?
        rendered
      end
    end

    # One file from an agent's knowledge directory, as it will be ingested.
    Document = Data.define(:source, :text)

    # One page from knowledge/urls.yaml. A page is a subscription rather than a copy.
    KnowledgeURL = Data.define(:url, :title, :description) do
      def initialize(url:, title: "", description: "")
        super
      end

      def declaration
        { url: url, title: (title unless title.empty?), description: (description unless description.empty?) }.compact
      end
    end

    # An agent written down as a directory.
    #
    #   agents/jean/
    #     agent.yaml          required: what the agent is called and what it runs on
    #     instructions.md
    #     guardrail.md
    #     skills/think.md
    #     knowledge/pricing.md
    #     knowledge/urls.yaml
    #     .agent_sync         written by Agent#sync: the fingerprint last synced, and when
    #
    # The fingerprint is taken exactly the way the Go and Python SDKs take it, so a stamp any
    # of them wrote is understood by the others.
    class Folder
      AGENT_FILE = "agent.yaml"
      STAMP_FILE = ".agent_sync"
      INSTRUCTIONS_FILE = "instructions.md"
      GUARDRAIL_FILE = "guardrail.md"
      SKILLS_DIR = "skills"
      KNOWLEDGE_DIR = "knowledge"
      URLS_FILE = "urls.yaml"

      # The extensions a knowledge directory is read from. A model looks things up in prose,
      # not in a binary.
      READABLE = %w[.md .mdx .txt .rst .yaml .yml].freeze

      SETTING_STRINGS = %w[name description mode stt tts sts voice llm subagent search greeting sandbox].freeze
      SETTING_LISTS = %w[plugins keyterms].freeze
      SETTINGS = (SETTING_STRINGS + SETTING_LISTS + %w[tags video]).freeze
      VIDEO_KEYS = %w[source max_frames].freeze

      attr_reader :path, :name, :declaration, :settings, :instructions, :guardrail, :skills,
                  :knowledge, :knowledge_urls

      # Reads an agent directory.
      #
      # agent.yaml is what makes a directory an agent, so it is required. Everything else is
      # optional.
      #
      # @raise [ConfigurationError] for a directory that is not an agent, or one that says
      #   something this cannot read.
      def self.load(path)
        new(path)
      end

      def initialize(path)
        @path = File.expand_path(path)
        raise ConfigurationError, "#{path} is not an agent directory" unless File.directory?(@path)

        declared = File.join(@path, AGENT_FILE)
        unless File.file?(declared)
          raise ConfigurationError, "#{path} has no #{AGENT_FILE}, so it is not an agent directory"
        end

        raw = File.read(declared)
        @declaration = raw.strip
        @settings = Folder.declare(raw, declared)
        @name = @settings["name"].to_s.empty? ? File.basename(@path) : @settings["name"]
        @instructions = optional(INSTRUCTIONS_FILE)
        @guardrail = optional(GUARDRAIL_FILE)
        @skills = load_skills
        @knowledge = load_knowledge
        @knowledge_urls = load_urls
      end

      # A fingerprint of the directory. The same files produce the same hash in every SDK.
      def fingerprint
        Folder.fingerprint(@declaration, @instructions, @guardrail, @skills, @knowledge, @knowledge_urls)
      end

      # The fingerprint the directory was last synced under, or nil when it never was or the
      # stamp cannot be read.
      def stamp
        recorded = JSON.parse(File.read(File.join(@path, STAMP_FILE)))
        recorded.is_a?(Hash) ? recorded["hash"] : nil
      rescue SystemCallError, JSON::ParserError
        nil
      end

      # Records what was synced and when, so a second sync can do nothing.
      def write_stamp(hash)
        Folder.write_stamp(@path, STAMP_FILE, hash)
      end

      # The stamp format, shared with a router folder's .router_sync.
      def self.write_stamp(directory, file, hash)
        synced_at = Time.now.utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        File.write(File.join(directory, file), "#{JSON.generate({ hash: hash, synced_at: synced_at })}\n")
      end

      # Ported from Go's fingerprint byte for byte: the bool and the float are written the
      # way Python prints them, which is what keeps the three SDKs' fingerprints the same.
      def self.fingerprint(declaration, instructions, guardrail, skills, knowledge, pages)
        digest = Digest::MD5.new
        digest << "#{declaration}\n#{instructions}\n#{guardrail}"
        skills.sort_by(&:name).each do |skill|
          digest << "\nskill:#{skill.name}\n#{skill.description}\n#{skill.instructions}" \
                    "#{skill.capture_video ? "True" : "False"}\n"
          digest << seconds(skill.deadline) if skill.deadline&.positive?
        end
        knowledge.sort_by(&:source).each do |document|
          digest << "\nknowledge:#{document.source}\n#{document.text}"
        end
        pages.each do |page|
          digest << "\nurl:#{page.url}\n#{page.title}\n#{page.description}"
        end
        digest.hexdigest
      end

      # Go's strconv.FormatFloat(s, 'f', -1, 64), with ".0" added the way Python writes a
      # whole float.
      def self.seconds(value)
        text = value.to_f.to_s
        text = format("%.17g", value.to_f).sub(/\.?0+\z/, "") if text.include?("e")
        text.include?(".") ? text : "#{text}.0"
      end

      # Reads agent.yaml. A key nobody knows is refused rather than dropped, since a
      # misspelled llm that goes quietly is a config running on a model the file does not
      # name.
      def self.declare(raw, file = AGENT_FILE)
        parsed = Psych.safe_load(raw, aliases: false) || {}
        raise ConfigurationError, "#{file} should be a mapping of settings" unless parsed.is_a?(Hash)

        unknown = parsed.keys.map(&:to_s) - SETTINGS
        unless unknown.empty?
          raise ConfigurationError, "#{file}: #{unknown.join(", ")} is not a setting; #{SETTINGS.join(", ")} are"
        end

        parsed.to_h do |key, value|
          key = key.to_s
          [key, setting(key, value, file)]
        end
      rescue Psych::Exception => e
        raise ConfigurationError, "#{file}: #{e.message}"
      end

      def self.setting(key, value, file)
        return value if value.nil?

        case key
        when *SETTING_STRINGS
          raise ConfigurationError, "#{file}: #{key} should be a string" if value.is_a?(Hash) || value.is_a?(Array)

          value.to_s
        when *SETTING_LISTS
          unless value.is_a?(Array) && value.none? { |each| each.is_a?(Hash) || each.is_a?(Array) }
            raise ConfigurationError, "#{file}: #{key} should be a list"
          end

          value.map(&:to_s)
        when "tags"
          raise ConfigurationError, "#{file}: tags should be a mapping" unless value.is_a?(Hash)

          value.to_h { |name, label| [name.to_s, label.to_s] }
        when "video" then video(value, file)
        end
      end

      def self.video(value, file)
        raise ConfigurationError, "#{file}: video should be a mapping" unless value.is_a?(Hash)

        unknown = value.keys.map(&:to_s) - VIDEO_KEYS
        raise ConfigurationError, "#{file}: video has no #{unknown.join(", ")}" unless unknown.empty?

        frames = value["max_frames"] || 0
        frames = 1 if frames.zero?
        unless frames.is_a?(Integer) && frames.between?(1, 8)
          raise ConfigurationError, "#{file}: video.max_frames must be an integer from 1 to 8"
        end

        { "source" => value["source"]&.to_s, "max_frames" => frames }.compact
      end

      # Reads a skill file: frontmatter between --- lines, then the instructions. The keys
      # read are name, description, capture_video and deadline, which is a Go duration such
      # as "30s", or a bare number of seconds.
      def self.parse_skill(name, content)
        frontmatter, body = cut_frontmatter(content)
        fields = { name: name, description: "", capture_video: false, deadline: nil }
        frontmatter.to_s.each_line do |line|
          line = line.strip
          next if line.empty? || line.start_with?("#")

          key, separator, value = line.partition(":")
          raise ConfigurationError, "#{line.inspect} is not a key and a value" if separator.empty?

          value = value.strip.gsub(/\A["']+|["']+\z/, "")
          case key.strip
          when "name" then fields[:name] = value
          when "description" then fields[:description] = value
          when "capture_video"
            raise ConfigurationError, "capture_video must be true or false" unless %w[true false].include?(value)

            fields[:capture_video] = value == "true"
          when "deadline" then fields[:deadline] = deadline(value)
          end
        end

        instructions = body.strip
        if fields[:description].empty?
          raise ConfigurationError, "a skill needs a description, since it is all the fast model sees"
        end
        if instructions.empty?
          raise ConfigurationError, "a skill needs instructions, since they are what the subagent answers under"
        end

        Skill.new(instructions: instructions, **fields)
      end

      UNITS = { "ns" => 1, "us" => 1_000, "µs" => 1_000, "μs" => 1_000, "ms" => 1_000_000,
                "s" => 1_000_000_000, "m" => 60_000_000_000, "h" => 3_600_000_000_000 }.freeze

      # A Go duration or a bare number of seconds, as seconds. Taken through whole
      # nanoseconds the way time.Duration holds it, so the fingerprint matches Go's.
      def self.deadline(value)
        nanoseconds =
          if value.match?(/\A[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\z/)
            (Float(value) * 1_000_000_000).to_i
          else
            parts = value.scan(/(\d+\.?\d*|\.\d+)(ns|us|µs|μs|ms|s|m|h)/)
            raise ConfigurationError, "#{value.inspect} is not a deadline" if parts.empty? || parts.join != value.delete("+")

            parts.sum { |number, unit| (Float(number) * UNITS.fetch(unit)).to_i }
          end
        nanoseconds / 1_000_000_000.0
      end

      def self.cut_frontmatter(content)
        trimmed = content.sub(/\A[\uFEFF \t\r\n]+/, "")
        return [nil, content] unless trimmed.start_with?("---")

        rest = trimmed.delete_prefix("---").sub(/\A[\r\n]+/, "")
        frontmatter, separator, body = rest.partition("\n---")
        return [nil, content] if separator.empty?

        [frontmatter, body.sub(/\A[-\r\n]+/, "")]
      end

      private

      def optional(file)
        full = File.join(@path, file)
        File.file?(full) ? File.read(full).strip : ""
      end

      def load_skills
        directory = File.join(@path, SKILLS_DIR)
        return [] unless File.directory?(directory)

        Dir.children(directory).sort.filter_map do |entry|
          file = File.join(directory, entry)
          next unless File.file?(file) && File.extname(entry) == ".md"

          Folder.parse_skill(File.basename(entry, ".md"), File.read(file))
        rescue ConfigurationError => e
          raise ConfigurationError, "#{file}: #{e.message}"
        end
      end

      def load_knowledge
        root = File.join(@path, KNOWLEDGE_DIR)
        return [] unless File.exist?(root)
        raise ConfigurationError, "#{root} is not a directory" unless File.directory?(root)

        declaration = File.join(root, URLS_FILE)
        Dir.glob("**/*", File::FNM_DOTMATCH, base: root).sort.filter_map do |relative|
          file = File.join(root, relative)
          next unless File.file?(file) && READABLE.include?(File.extname(relative).downcase)
          # Only the declaration at the root is not a document; deeper, urls.yaml is prose.
          next if file == declaration

          text = File.read(file)
          Document.new(relative, text) unless text.strip.empty?
        end
      end

      # A bad url is refused here rather than when it is subscribed, since a directory that
      # cannot be turned into a knowledge base is worth hearing about before anything is
      # written.
      def load_urls
        file = File.join(@path, KNOWLEDGE_DIR, URLS_FILE)
        return [] unless File.file?(file)

        pages = Psych.safe_load(File.read(file), aliases: false) || []
        raise ConfigurationError, "#{file}: should be a list of pages" unless pages.is_a?(Array)

        pages.map { |page| Folder.page(page, file) }
      rescue Psych::Exception => e
        raise ConfigurationError, "#{file}: #{e.message}"
      end

      # A page is a url on its own, or a mapping naming one alongside what it is. Unknown keys
      # are refused, so a misspelt one is reported rather than dropped.
      def self.page(entry, file)
        page =
          case entry
          when String then KnowledgeURL.new(url: entry)
          when Hash
            unknown = entry.keys.map(&:to_s) - %w[url title description]
            unless unknown.empty?
              raise ConfigurationError,
                    "#{file}: #{unknown.join(", ")} is not something a page says; url, title and description are"
            end

            KnowledgeURL.new(url: entry["url"].to_s, title: entry["title"].to_s,
                             description: entry["description"].to_s)
          else
            raise ConfigurationError, "#{file}: a page is a url, or a mapping naming one"
          end
        unless page.url.start_with?("http://", "https://")
          raise ConfigurationError, "#{file}: #{page.url.inspect} is not an http or https url"
        end

        page
      end
    end
  end
end
