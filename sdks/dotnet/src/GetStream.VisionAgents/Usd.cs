using System.Globalization;

namespace GetStream.VisionAgents;

/// <summary>
/// An amount of dollars, held exactly as the millionths of a dollar the router counts in.
/// </summary>
/// <remarks>
/// There is no conversion from a <see cref="double"/> on purpose: 0.1 + 0.2 is not a price.
/// <see cref="Parse"/> reads one written out, such as <c>"12.50"</c>, and
/// <see cref="FromDecimal"/> takes a <see cref="decimal"/>. What the router answers is micros
/// too, so <c>Usd.FromMicros(limit.LimitMicros)</c> is a limit in dollars.
/// <code>
/// var limit = Usd.Dollars(20_000);
/// var price = Usd.Cents(3);
/// </code>
/// </remarks>
public readonly struct Usd : IEquatable<Usd>, IComparable<Usd>
{
    /// <summary>Millionths of a dollar in one dollar.</summary>
    public const long MicrosPerDollar = 1_000_000;

    private const long MicrosPerCent = 10_000;

    private Usd(long micros) => Micros = micros;

    /// <summary>Nothing.</summary>
    public static Usd Zero => default;

    /// <summary>The amount in millionths of a dollar, as the router's fields want it.</summary>
    public long Micros { get; }

    /// <summary>An amount the router answered, in millionths of a dollar.</summary>
    public static Usd FromMicros(long micros) => new(micros);

    /// <summary>Whole dollars.</summary>
    /// <exception cref="OverflowException">More than the router can count.</exception>
    public static Usd Dollars(long dollars) => new(checked(dollars * MicrosPerDollar));

    /// <summary>Hundredths of a dollar.</summary>
    /// <exception cref="OverflowException">More than the router can count.</exception>
    public static Usd Cents(long cents) => new(checked(cents * MicrosPerCent));

    /// <summary>Dollars as a decimal, such as <c>12.50m</c>.</summary>
    /// <exception cref="FormatException">It is finer than a millionth of a dollar.</exception>
    /// <exception cref="OverflowException">More than the router can count.</exception>
    public static Usd FromDecimal(decimal dollars)
    {
        var micros = dollars * MicrosPerDollar;
        if (micros != decimal.Truncate(micros))
        {
            throw new FormatException($"{dollars.ToString(CultureInfo.InvariantCulture)} is finer than a millionth of a dollar");
        }
        return new(decimal.ToInt64(micros));
    }

    /// <summary>
    /// Reads an amount written out: <c>"12"</c>, <c>"12.5"</c>, <c>"$12.50"</c>, <c>"-3.25"</c>
    /// or <c>"0.000300"</c>.
    /// </summary>
    /// <remarks>
    /// Anything finer than a millionth of a dollar is refused, since the router cannot count
    /// it, and so is anything but plain digits: no thousands separators, no exponent.
    /// </remarks>
    /// <exception cref="FormatException">It is not an amount of dollars the router can count.</exception>
    public static Usd Parse(string written) =>
        Read(written, out var amount) is { } refused ? throw new FormatException(refused) : amount;

    /// <summary>Reads an amount written out, as <see cref="Parse"/> does, or says it could not.</summary>
    public static bool TryParse(string? written, out Usd amount) => Read(written, out amount) is null;

    /// <summary>The amount as a decimal number of dollars, exactly.</summary>
    public decimal ToDecimal() => Micros / (decimal)MicrosPerDollar;

    /// <summary>
    /// The amount as dollars, with cents always and finer digits only when it has them:
    /// <c>$20000.00</c>, <c>$12.50</c>, <c>$0.0003</c>. <see cref="Parse"/> reads it back.
    /// </summary>
    public override string ToString()
    {
        var magnitude = Micros < 0 ? (ulong)(-(Micros + 1)) + 1 : (ulong)Micros;
        var digits = (magnitude % MicrosPerDollar).ToString("D6", CultureInfo.InvariantCulture).TrimEnd('0');
        return $"{(Micros < 0 ? "-" : "")}${magnitude / MicrosPerDollar}.{digits.PadRight(2, '0')}";
    }

    /// <inheritdoc/>
    public bool Equals(Usd other) => Micros == other.Micros;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Usd other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => Micros.GetHashCode();

    /// <inheritdoc/>
    public int CompareTo(Usd other) => Micros.CompareTo(other.Micros);

    /// <summary>Whether two amounts are the same.</summary>
    public static bool operator ==(Usd left, Usd right) => left.Equals(right);

    /// <summary>Whether two amounts differ.</summary>
    public static bool operator !=(Usd left, Usd right) => !left.Equals(right);

    /// <summary>Whether the left amount is less.</summary>
    public static bool operator <(Usd left, Usd right) => left.Micros < right.Micros;

    /// <summary>Whether the left amount is more.</summary>
    public static bool operator >(Usd left, Usd right) => left.Micros > right.Micros;

    /// <summary>Whether the left amount is no more.</summary>
    public static bool operator <=(Usd left, Usd right) => left.Micros <= right.Micros;

    /// <summary>Whether the left amount is no less.</summary>
    public static bool operator >=(Usd left, Usd right) => left.Micros >= right.Micros;

    /// <summary>The two amounts together.</summary>
    public static Usd operator +(Usd left, Usd right) => new(checked(left.Micros + right.Micros));

    /// <summary>The left amount less the right.</summary>
    public static Usd operator -(Usd left, Usd right) => new(checked(left.Micros - right.Micros));

    /// <summary>The amount below zero, or back above it.</summary>
    public static Usd operator -(Usd amount) => new(checked(-amount.Micros));

    /// <summary>The amount, so many times.</summary>
    public static Usd operator *(Usd amount, long times) => new(checked(amount.Micros * times));

    /// <summary>The amount, so many times.</summary>
    public static Usd operator *(long times, Usd amount) => amount * times;

    private static string? Read(string? written, out Usd amount)
    {
        amount = default;
        var text = (written ?? "").Trim();
        var negative = text.StartsWith('-');
        if (negative)
        {
            text = text[1..];
        }
        if (text.StartsWith('$'))
        {
            text = text[1..];
        }
        var dot = text.IndexOf('.');
        var whole = dot < 0 ? text : text[..dot];
        var fraction = dot < 0 ? "" : text[(dot + 1)..];
        if ((whole == "" && fraction == "") || !Digits(whole) || !Digits(fraction) || (dot >= 0 && fraction == ""))
        {
            return $"\"{written}\" is not an amount of dollars";
        }
        if (fraction.Length > 6)
        {
            if (fraction[6..].Trim('0') != "")
            {
                return $"\"{written}\" is finer than a millionth of a dollar";
            }
            fraction = fraction[..6];
        }
        var micros = fraction == "" ? 0 : long.Parse(fraction.PadRight(6, '0'), CultureInfo.InvariantCulture);
        if (!long.TryParse(whole == "" ? "0" : whole, NumberStyles.None, CultureInfo.InvariantCulture, out var dollars)
            || dollars > (long.MaxValue - micros) / MicrosPerDollar)
        {
            return $"\"{written}\" is too large an amount";
        }
        var total = (dollars * MicrosPerDollar) + micros;
        amount = new(negative ? -total : total);
        return null;
    }

    private static bool Digits(string text) => text.All(char.IsAsciiDigit);
}
