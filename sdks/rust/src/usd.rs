use std::fmt;
use std::str::FromStr;

use crate::error::Error;

const MICROS_PER_DOLLAR: i64 = 1_000_000;
const MICROS_PER_CENT: i64 = 10_000;

/// An amount of dollars, held exactly as the millionths of a dollar the router counts in.
///
/// ```
/// use vision_agents::Usd;
///
/// assert_eq!(Usd::dollars(20_000).to_string(), "$20000.00");
/// assert_eq!(Usd::cents(1_250), "12.50".parse().unwrap());
/// assert_eq!(Usd::from_micros(300).to_string(), "$0.0003");
/// ```
///
/// There is no constructor from a float on purpose: 0.1 + 0.2 is not a price. What the router
/// answers is micros too, so `Usd::from_micros(limit.limit_micros)` is a limit in dollars.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Usd(i64);

impl Usd {
    /// Panics when the amount cannot be counted in micros, past $9.2 trillion.
    pub const fn dollars(dollars: i64) -> Usd {
        match dollars.checked_mul(MICROS_PER_DOLLAR) {
            Some(micros) => Usd(micros),
            None => panic!("too many dollars to count in millionths of a dollar"),
        }
    }

    /// Panics when the amount cannot be counted in micros.
    pub const fn cents(cents: i64) -> Usd {
        match cents.checked_mul(MICROS_PER_CENT) {
            Some(micros) => Usd(micros),
            None => panic!("too many cents to count in millionths of a dollar"),
        }
    }

    /// Millionths of a dollar, the router's unit.
    pub const fn from_micros(micros: i64) -> Usd {
        Usd(micros)
    }

    /// The amount in millionths of a dollar, as the router's fields want it.
    pub const fn micros(self) -> i64 {
        self.0
    }
}

/// Dollars with cents always and finer digits only when there are some: `$20000.00`,
/// `$12.50`, `$0.0003`. Parsing reads it back.
impl fmt::Display for Usd {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.0 < 0 { "-" } else { "" };
        let magnitude = self.0.unsigned_abs();
        let per_dollar = MICROS_PER_DOLLAR as u64;
        let (whole, fraction) = (magnitude / per_dollar, magnitude % per_dollar);
        let digits = format!("{fraction:06}");
        let digits = digits.trim_end_matches('0');
        write!(formatter, "{sign}${whole}.{digits:0<2}")
    }
}

/// Reads an amount of dollars written out: `12`, `12.5`, `$12.50`, `-3.25` or `0.000300`.
/// Anything finer than a millionth of a dollar is refused, since the router cannot count it,
/// and so is anything but plain digits: no thousands separators, no exponent.
impl FromStr for Usd {
    type Err = Error;

    fn from_str(written: &str) -> Result<Usd, Error> {
        let refused = |why: &str| Error::configuration(format!("{written:?} {why}"));
        let text = written.trim();
        let (negative, text) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let text = text.strip_prefix('$').unwrap_or(text);
        let (whole, fraction, dotted) = match text.split_once('.') {
            Some((whole, fraction)) => (whole, fraction, true),
            None => (text, "", false),
        };
        let digits = |part: &str| part.bytes().all(|b| b.is_ascii_digit());
        if (whole.is_empty() && fraction.is_empty())
            || !digits(whole)
            || !digits(fraction)
            || (dotted && fraction.is_empty())
        {
            return Err(refused("is not an amount of dollars"));
        }
        let fraction = match fraction.get(6..) {
            Some(finer) if finer.bytes().any(|b| b != b'0') => {
                return Err(refused("is finer than a millionth of a dollar"));
            }
            Some(_) => &fraction[..6],
            None => fraction,
        };
        let dollars: i64 = if whole.is_empty() {
            0
        } else {
            whole
                .parse()
                .map_err(|_| refused("is too large an amount"))?
        };
        let micros: i64 = if fraction.is_empty() {
            0
        } else {
            format!("{fraction:0<6}").parse().unwrap_or_default()
        };
        let amount = dollars
            .checked_mul(MICROS_PER_DOLLAR)
            .and_then(|whole| whole.checked_add(micros))
            .ok_or_else(|| refused("is too large an amount"))?;
        Ok(Usd(if negative { -amount } else { amount }))
    }
}
