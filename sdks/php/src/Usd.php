<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;

/**
 * An amount of dollars, held exactly as the millionths of a dollar the router counts in.
 *
 *     $limit = Usd::dollars(20_000);
 *     $price = Usd::cents(3);
 *     $amount = Usd::parse('12.50');   // from a form or a config file
 *
 * There is no constructor from a float on purpose: 0.1 + 0.2 is not a price. What the router
 * answers is micros too, so `Usd::micros($limit->limitMicros)` is a limit in dollars.
 */
final readonly class Usd implements \Stringable
{
    public const int MICROS_PER_DOLLAR = 1_000_000;
    public const int MICROS_PER_CENT = 10_000;

    private function __construct(public int $micros)
    {
    }

    public static function micros(int $micros): self
    {
        return new self($micros);
    }

    public static function cents(int $cents): self
    {
        return new self(self::scaled($cents, self::MICROS_PER_CENT));
    }

    public static function dollars(int $dollars): self
    {
        return new self(self::scaled($dollars, self::MICROS_PER_DOLLAR));
    }

    /**
     * Reads an amount of dollars written out: "12", "12.5", "$12.50", "-3.25" or "0.000300".
     * Anything finer than a millionth of a dollar, which the router cannot count, is refused,
     * and so is anything that is not plain digits: no thousands separators, no exponent.
     */
    public static function parse(string $written): self
    {
        if (preg_match('/\A(-)?\$?([0-9]*)(?:\.([0-9]+))?\z/', trim($written), $match) !== 1 || ($match[2] === '' && ($match[3] ?? '') === '')) {
            throw new ConfigurationException(var_export($written, true) . ' is not an amount of dollars');
        }
        $fraction = $match[3] ?? '';
        if (strlen($fraction) > 6) {
            if (trim(substr($fraction, 6), '0') !== '') {
                throw new ConfigurationException(var_export($written, true) . ' is finer than a millionth of a dollar');
            }
            $fraction = substr($fraction, 0, 6);
        }
        $whole = ltrim($match[2], '0');
        $micros = (int) str_pad($fraction, 6, '0');
        if (strlen($whole) > 13 || ($whole !== '' && (int) $whole > intdiv(PHP_INT_MAX - $micros, self::MICROS_PER_DOLLAR))) {
            throw new ConfigurationException(var_export($written, true) . ' is too large an amount');
        }
        $amount = (int) $whole * self::MICROS_PER_DOLLAR + $micros;
        return new self($match[1] === '-' ? -$amount : $amount);
    }

    /**
     * The amount as dollars, with cents always and finer digits only when it has them:
     * $20000.00, $12.50, $0.0003. `parse` reads it back.
     */
    public function __toString(): string
    {
        $whole = abs(intdiv($this->micros, self::MICROS_PER_DOLLAR));
        $fraction = abs($this->micros % self::MICROS_PER_DOLLAR);
        $digits = str_pad(rtrim(sprintf('%06d', $fraction), '0'), 2, '0');
        return ($this->micros < 0 ? '-' : '') . '$' . $whole . '.' . $digits;
    }

    private static function scaled(int $amount, int $unit): int
    {
        if ($amount > intdiv(PHP_INT_MAX, $unit) || $amount < intdiv(PHP_INT_MIN, $unit)) {
            throw new ConfigurationException("{$amount} is too large an amount");
        }
        return $amount * $unit;
    }
}
