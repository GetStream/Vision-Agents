import { ConfigurationError } from "./errors.js";

const MICROS_PER_DOLLAR = 1_000_000n;
const MICROS_PER_CENT = 10_000n;
const INT64_MAX = 2n ** 63n - 1n;
const WRITTEN = /^(-?)\$?(\d*)(?:\.(\d*))?$/;

/**
 * An amount of dollars, held exactly as the millionths of a dollar the router counts in.
 *
 * ```ts
 * USD.dollars(20_000);  // $20000.00
 * USD.cents(1_250);     // $12.50
 * USD.parse("0.0003");  // $0.0003, what one short LLM call can cost
 * ```
 *
 * There is no constructor from a fractional number on purpose: 0.1 + 0.2 is not a price.
 * `parse` reads one written out, and `fromMicros` reads what the router answers, such as a
 * limit's `limit_micros`.
 */
export class USD {
  /** The amount in millionths of a dollar. */
  readonly micros: bigint;

  private constructor(micros: bigint) {
    if (micros > INT64_MAX || micros < -INT64_MAX - 1n) {
      throw new ConfigurationError(`${micros} micros is too large an amount`);
    }
    this.micros = micros;
  }

  /** Whole dollars. */
  static dollars(dollars: number | bigint): USD {
    return new USD(whole(dollars, "dollars") * MICROS_PER_DOLLAR);
  }

  /** Whole cents. */
  static cents(cents: number | bigint): USD {
    return new USD(whole(cents, "cents") * MICROS_PER_CENT);
  }

  /** Millionths of a dollar, which is what the router answers amounts in. */
  static fromMicros(micros: number | bigint): USD {
    return new USD(whole(micros, "micros"));
  }

  /**
   * An amount of dollars written out: "12", "12.5", "$12.50", "-3.25" or "0.000300". It
   * refuses anything finer than a millionth of a dollar, which the router cannot count, and
   * anything that is not plain digits: no thousands separators, no exponent.
   */
  static parse(written: string): USD {
    const match = WRITTEN.exec(written.trim());
    const [, sign = "", dollars = "", fraction] = match ?? [];
    if (!match || (dollars === "" && !fraction) || fraction === "") {
      throw new ConfigurationError(`${JSON.stringify(written)} is not an amount of dollars`);
    }
    const digits = fraction ?? "";
    if (/[^0]/.test(digits.slice(6))) {
      throw new ConfigurationError(`${JSON.stringify(written)} is finer than a millionth of a dollar`);
    }
    const micros = BigInt(dollars || "0") * MICROS_PER_DOLLAR + BigInt(digits.slice(0, 6).padEnd(6, "0"));
    return new USD(sign ? -micros : micros);
  }

  /** The amount as the router's `*_micros` fields want it. */
  toWire(): number {
    if (this.micros > BigInt(Number.MAX_SAFE_INTEGER) || this.micros < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new ConfigurationError(`${this.toString()} is more than a JSON number holds exactly`);
    }
    return Number(this.micros);
  }

  /**
   * The amount as dollars, with cents always and finer digits only when it has them:
   * $20000.00, $12.50, $0.0003. `parse` reads it back.
   */
  toString(): string {
    const negative = this.micros < 0n;
    const magnitude = negative ? -this.micros : this.micros;
    const fraction = (magnitude % MICROS_PER_DOLLAR).toString().padStart(6, "0").replace(/0+$/, "");
    return `${negative ? "-" : ""}$${magnitude / MICROS_PER_DOLLAR}.${fraction.padEnd(2, "0")}`;
  }

  /** The same amount: two `USD` are never `===`. */
  equals(other: USD): boolean {
    return this.micros === other.micros;
  }
}

/** What an amount may be passed as: a `USD`, or dollars written out for `USD.parse`. */
export type Amount = USD | string;

/** The amount as a `USD`, reading one written out. */
export function amountOf(amount: Amount): USD {
  if (amount instanceof USD) {
    return amount;
  }
  if (typeof amount === "string") {
    return USD.parse(amount);
  }
  throw new ConfigurationError(
    `an amount is a USD or a string of dollars, not ${typeof amount === "number" ? "a number" : typeof amount}`,
  );
}

function whole(value: number | bigint, unit: string): bigint {
  if (typeof value === "bigint") {
    return value;
  }
  if (!Number.isSafeInteger(value)) {
    throw new ConfigurationError(
      `${value} is not a whole number of ${unit}; write a fraction out, as USD.parse("12.50")`,
    );
  }
  return BigInt(value);
}
