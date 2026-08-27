"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  Button,
  Empty,
  Failure,
  Field,
  inputStyle,
  PageHeading,
  Panel,
} from "@/components/ui";
import {
  router,
  type AvailableNumber,
  type NumberSearchResult,
  type PhoneNumber,
} from "@/lib/router";

function dollars(micros: number): string {
  return `$${(micros / 1_000_000).toFixed(2)}`;
}

export default function Telephony() {
  const client = useQueryClient();
  const numbers = useQuery({ queryKey: ["numbers"], queryFn: router.numbers });
  const refresh = () => client.invalidateQueries({ queryKey: ["numbers"] });

  const [adding, setAdding] = useState(false);

  return (
    <>
      <PageHeading
        title="Telephony"
        description="The numbers you hold, and what happens when somebody rings one."
        action={
          <Button onClick={() => setAdding((open) => !open)}>
            {adding ? "Close" : "Add a number"}
          </Button>
        }
      />

      {adding ? (
        <AddNumber
          onBought={() => {
            refresh();
            setAdding(false);
          }}
        />
      ) : null}

      {numbers.isError ? (
        <div className="mt-6">
          <Failure error={numbers.error} />
        </div>
      ) : null}

      <Panel title="Your numbers" className="mt-6">
        {numbers.data?.length === 0 ? (
          <Empty>
            No numbers yet. Nobody can ring an agent until one is attached.
          </Empty>
        ) : null}
        <ul>
          {(numbers.data ?? []).map((number) => (
            <NumberRow key={number.e164} number={number} onChanged={refresh} />
          ))}
        </ul>
      </Panel>
    </>
  );
}

function NumberRow({
  number,
  onChanged,
}: {
  number: PhoneNumber;
  onChanged: () => void;
}) {
  const attach = useMutation({
    mutationFn: () => router.attachNumber(number.e164, {}),
    onSuccess: onChanged,
  });
  const release = useMutation({
    mutationFn: () => router.releaseNumber(number.e164),
    onSuccess: onChanged,
  });

  const attached = Boolean(number.stream_trunk_id);
  const released = Boolean(number.released_at);

  return (
    <li className="flex items-center gap-4 border-b border-line px-4 py-3 last:border-0">
      <div className="min-w-0 flex-1">
        <div className="font-mono font-medium">{number.e164}</div>
        <div className="mt-1 flex flex-wrap gap-1.5 text-xs text-muted">
          <span className="rounded-md border border-line px-1.5 py-0.5">
            {number.vendor}
          </span>
          <span className="rounded-md border border-line px-1.5 py-0.5">
            {number.country}
          </span>
          <span className="rounded-md border border-line px-1.5 py-0.5">
            {dollars(number.monthly_cost_micros)}/mo
          </span>
          {number.capabilities.map((capability) => (
            <span
              key={capability}
              className="rounded-md border border-line px-1.5 py-0.5"
            >
              {capability}
            </span>
          ))}
          {released ? (
            <span className="rounded-md border border-line px-1.5 py-0.5 text-red-600">
              released
            </span>
          ) : (
            <span
              className={`rounded-md border px-1.5 py-0.5 ${
                attached
                  ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-600"
                  : "border-amber-500/20 bg-amber-500/10 text-amber-600"
              }`}
            >
              {attached ? "reaches an agent" : "not attached"}
            </span>
          )}
        </div>
        {attach.error ? (
          <p className="mt-2 text-xs text-red-600">
            {(attach.error as Error).message}
          </p>
        ) : null}
      </div>
      {released ? null : (
        <div className="flex shrink-0 gap-2">
          {attached ? null : (
            <Button
              variant="quiet"
              onClick={() => attach.mutate()}
              disabled={attach.isPending}
            >
              {attach.isPending ? "Attaching…" : "Attach"}
            </Button>
          )}
          <Button
            variant="danger"
            onClick={() => release.mutate()}
            disabled={release.isPending}
          >
            Release
          </Button>
        </div>
      )}
    </li>
  );
}

function AddNumber({ onBought }: { onBought: () => void }) {
  const [country, setCountry] = useState("US");
  const [areaCode, setAreaCode] = useState("");
  const [contains, setContains] = useState("");
  const [found, setFound] = useState<NumberSearchResult | null>(null);

  const search = useMutation({
    mutationFn: () =>
      router.searchNumbers({
        country,
        area_code: areaCode || undefined,
        contains: contains || undefined,
      }),
    onSuccess: setFound,
  });

  const buy = useMutation({
    mutationFn: (offer: AvailableNumber) =>
      router.buyNumber({
        e164: offer.e164,
        vendor: offer.vendor,
        country: offer.country,
      }),
    onSuccess: onBought,
  });

  return (
    <Panel title="Buy a number">
      <form
        className="flex flex-wrap items-end gap-3 px-4 py-4"
        onSubmit={(event) => {
          event.preventDefault();
          search.mutate();
        }}
      >
        <div className="w-24">
          <Field label="Country">
            <input
              className={inputStyle}
              value={country}
              onChange={(event) =>
                setCountry(event.target.value.toUpperCase().slice(0, 2))
              }
              required
            />
          </Field>
        </div>
        <div className="w-28">
          <Field label="Area code">
            <input
              className={inputStyle}
              value={areaCode}
              onChange={(event) => setAreaCode(event.target.value)}
            />
          </Field>
        </div>
        <div className="w-36">
          <Field label="Contains">
            <input
              className={inputStyle}
              value={contains}
              onChange={(event) => setContains(event.target.value)}
            />
          </Field>
        </div>
        <Button type="submit" disabled={search.isPending}>
          {search.isPending ? "Searching…" : "Search every vendor"}
        </Button>
      </form>

      {search.error ? (
        <div className="px-4 pb-4">
          <Failure error={search.error} />
        </div>
      ) : null}

      {found ? (
        <div className="border-t border-line">
          {found.skipped.length ? (
            // A search that reached two of eight vendors found what two vendors had, and
            // deciding whether to buy needs to know which.
            <p className="border-b border-line px-4 py-2 text-xs text-muted">
              Not asked:{" "}
              {found.skipped
                .map((skipped) => `${skipped.vendor} (${skipped.reason})`)
                .join(", ")}
            </p>
          ) : null}
          {found.numbers.length === 0 ? (
            <Empty>No vendor is offering a number matching that.</Empty>
          ) : (
            <ul>
              {found.numbers.map((offer) => (
                <li
                  key={`${offer.vendor}-${offer.e164}`}
                  className="flex items-center gap-4 border-b border-line px-4 py-2 text-sm last:border-0"
                >
                  <span className="flex-1 font-mono">{offer.e164}</span>
                  <span className="text-muted">{offer.vendor}</span>
                  <span className="text-muted">
                    {[offer.locality, offer.region].filter(Boolean).join(", ") ||
                      offer.country}
                  </span>
                  <span className="w-20 text-right tabular-nums">
                    {offer.monthly_cost_micros
                      ? `${dollars(offer.monthly_cost_micros)}/mo`
                      : "—"}
                  </span>
                  <Button
                    variant="quiet"
                    onClick={() => buy.mutate(offer)}
                    disabled={buy.isPending}
                  >
                    Buy
                  </Button>
                </li>
              ))}
            </ul>
          )}
          {buy.error ? (
            <div className="px-4 py-3">
              <Failure error={buy.error} />
            </div>
          ) : null}
        </div>
      ) : null}
    </Panel>
  );
}
