# frozen_string_literal: true

module GetStream
  module VisionAgents
    # An amount of dollars, held exactly as the millionths of a dollar the router counts in.
    #
    #   GetStream::VisionAgents.usd(20_000)   # $20000.00
    #   GetStream::VisionAgents.usd("12.50")  # from a form or a config file
    #   GetStream::VisionAgents::USD.micros(limit["limit_micros"])
    class USD
      include Comparable

      MICROS_PER_DOLLAR = 1_000_000
      DECIMAL = /\A-?\d+(?:\.\d{1,6})?\z/

      attr_reader :micros

      # The amount the router answered, in millionths of a dollar.
      def self.micros(micros)
        raise ConfigurationError, "#{micros.inspect} is not a whole number of micros" unless micros.is_a?(Integer)

        new(micros)
      end

      # Dollars from text such as "12.50", exactly, or ConfigurationError saying why they are
      # not an amount.
      def self.parse(text)
        written = text.to_s.strip.delete("_")
        unless DECIMAL.match?(written)
          raise ConfigurationError, "#{text.inspect} is not an amount of dollars to the millionth"
        end

        dollars(Rational(written))
      end

      # Dollars from an Integer, a Rational, a String or a USD. A Float is refused: 0.1 + 0.2
      # is not a price.
      def self.of(amount)
        case amount
        when USD then amount
        when Integer, Rational then dollars(amount)
        when String then parse(amount)
        else
          raise ConfigurationError,
                "an amount is an Integer, a Rational or a String of dollars, or a USD, not a " \
                "#{amount.class}#{": a Float is not exact" if amount.is_a?(Float)}"
        end
      end

      def self.dollars(amount)
        exact = amount * MICROS_PER_DOLLAR
        raise ConfigurationError, "#{amount} dollars is finer than a millionth of a dollar" if exact.denominator != 1

        new(exact.to_i)
      end
      private_class_method :new, :dollars

      def initialize(micros)
        @micros = micros
        freeze
      end

      def negative?
        @micros.negative?
      end

      def +(other)
        USD.micros(@micros + USD.of(other).micros)
      end

      def -(other)
        USD.micros(@micros - USD.of(other).micros)
      end

      def <=>(other)
        @micros <=> other.micros if other.is_a?(USD)
      end

      def eql?(other)
        other.is_a?(USD) && @micros == other.micros
      end

      def hash
        @micros.hash
      end

      def to_r
        Rational(@micros, MICROS_PER_DOLLAR)
      end

      # Cents always, and the millionths only when there are any: $20000.00, $0.000125.
      def to_s
        whole, part = @micros.abs.divmod(MICROS_PER_DOLLAR)
        "#{"-" if negative?}$#{whole}.#{format("%06d", part).sub(/0+\z/, "").ljust(2, "0")}"
      end

      def inspect
        "#<#{self.class.name} #{self}>"
      end
    end

    # Dollars, exactly: see USD.of.
    def self.usd(amount)
      USD.of(amount)
    end
  end
end
