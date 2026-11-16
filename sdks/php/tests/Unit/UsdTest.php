<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Usd;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class UsdTest extends TestCase
{
    public function testUnitsAreMicros(): void
    {
        self::assertSame(20_000_000_000, Usd::dollars(20_000)->micros);
        self::assertSame(30_000, Usd::cents(3)->micros);
        self::assertSame(300, Usd::micros(300)->micros);
    }

    /**
     * @return iterable<string, array{string, int}>
     */
    public static function written(): iterable
    {
        yield 'whole' => ['12', 12_000_000];
        yield 'one decimal' => ['12.5', 12_500_000];
        yield 'dollar sign' => ['$12.50', 12_500_000];
        yield 'negative' => ['-3.25', -3_250_000];
        yield 'micros' => ['0.000300', 300];
        yield 'no whole part' => ['.5', 500_000];
        yield 'trailing zeros past micros' => ['1.1234560', 1_123_456];
        yield 'padded' => [' 7 ', 7_000_000];
        yield 'leading zeros' => ['007.01', 7_010_000];
        yield 'largest' => ['9223372036854.775807', PHP_INT_MAX];
    }

    #[DataProvider('written')]
    public function testParseReadsWhatIsWrittenExactly(string $written, int $micros): void
    {
        self::assertSame($micros, Usd::parse($written)->micros);
    }

    /**
     * @return iterable<string, array{string}>
     */
    public static function notAmounts(): iterable
    {
        foreach (['', '.', '12.', 'abc', '1,000', '1e3', '$-3', '--1', '1.0000001', '9223372036854.775808', '99999999999999999999'] as $written) {
            yield var_export($written, true) => [$written];
        }
    }

    #[DataProvider('notAmounts')]
    public function testParseRefusesWhatIsNotAnExactAmount(string $written): void
    {
        $this->expectException(ConfigurationException::class);
        Usd::parse($written);
    }

    public function testPrintsCentsAlwaysAndFinerDigitsWhenThereAreAny(): void
    {
        self::assertSame('$20000.00', (string) Usd::dollars(20_000));
        self::assertSame('$12.50', (string) Usd::parse('12.5'));
        self::assertSame('$0.0003', (string) Usd::micros(300));
        self::assertSame('-$3.25', (string) Usd::parse('-3.25'));
        self::assertSame('-$9223372036854.775808', (string) Usd::micros(PHP_INT_MIN));
        self::assertSame('$12.50', (string) Usd::parse((string) Usd::parse('12.50')));
    }

    public function testTooManyDollarsAreRefusedRatherThanWrapped(): void
    {
        $this->expectException(ConfigurationException::class);
        Usd::dollars(intdiv(PHP_INT_MAX, 1_000_000) + 1);
    }
}
