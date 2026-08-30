import { Term, Tooltip } from '../components/tooltip'
import { Empty, Panel, Pill } from '../components/ui'
import { summarizeByAdapter } from '../lib/derive'
import { bytes, count, percent, seconds } from '../lib/format'
import type { AdapterInventory, RequestDoneRecord } from '../lib/records'

/**
 * LoRA adapters: what is available, and what each one served.
 *
 * The two halves come from different places and are both needed. The inventory is reported by the
 * engine, so an adapter that has taken no traffic still appears. Usage is derived from completed
 * request records, so it survives replay. An adapter with no rows is discoverable and selectable
 * without ever having been used, which is exactly the case a traffic-only view would hide.
 *
 * The pool is unbounded and is not what costs VRAM; the resident slot count is. `count` adapters
 * are selectable, `slots` of them are on the device at any moment, and the engine swaps between
 * them at admission.
 */
export function AdaptersPanel({
  inventory,
  requests,
}: {
  inventory: AdapterInventory | undefined
  requests: RequestDoneRecord[]
}) {
  const usage = summarizeByAdapter(requests)
  const used = new Map(usage.map((entry) => [entry.name, entry]))
  const pooled = inventory?.names ?? []
  // Residency can only be asserted when the engine actually reported its bank. Without an
  // inventory the correct statement is "unknown", not "not loaded".
  const known = inventory !== undefined

  // Union of what is loaded and what appears in the records: a replayed log may name an adapter
  // this server no longer registers, and a live server may register one with no traffic.
  const names = [
    '',
    ...pooled,
    ...usage.map((entry) => entry.name).filter((name) => name !== '' && !pooled.includes(name)),
  ]
  const rows = names.filter((name, index) => names.indexOf(name) === index)

  if (pooled.length === 0 && usage.every((entry) => entry.name === '')) {
    return (
      <Panel title="Adapters" note="LoRA">
        <Empty>no adapters discovered</Empty>
      </Panel>
    )
  }

  const slots = inventory?.slots ?? inventory?.count ?? 0
  const perAdapter = inventory && slots > 0 ? inventory.device_bytes / slots : 0

  return (
    <Panel
      title="Adapters"
      className="panel--wide"
      note={
        inventory && inventory.count > 0 ? (
          <>
            {inventory.count} pooled · rank {inventory.rank} ·{' '}
            <Tooltip
              title="Adapter bank"
              body={`${inventory.count} adapter(s) are selectable; ${slots} slot(s) at ${bytes(
                perAdapter,
              )} each are device-resident. The bank is committed at startup in its own device arena,
 before KV capacity is resolved, and the engine stages an adapter into the least recently used
 free slot when a request selects one that is not resident.`}
              className="tip--term"
            >
              {slots} resident · {bytes(inventory.device_bytes)} vram
            </Tooltip>
          </>
        ) : known ? (
          'none discovered'
        ) : (
          // A schema-14 log and a pre-inventory engine both land here: usage is derivable, the
          // resident bank is not.
          'usage only · no inventory reported'
        )
      }
    >
      <table className="table">
        <thead>
          <tr>
            <th>adapter</th>
            <th className="numeric">reqs</th>
            <th className="numeric">gen</th>
            <th className="numeric">
              <Term k="ttft">ttft p50</Term>
            </th>
            <th className="numeric">
              <Term k="decodePerRequest">tok/s</Term>
            </th>
            <th className="numeric">
              <Term k="prefillAvoided">reused</Term>
            </th>
            <th className="numeric">
              <Term k="mtpAccept">mtp</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((name) => {
            const entry = used.get(name)
            // In the pool, not in a slot: the panel deliberately does not track slot occupancy,
            // which changes between polls and is not actionable.
            const inPool = name === '' || pooled.includes(name)
            const absent = known && !inPool
            return (
              <tr key={name || '<base>'}>
                <td className="emphasis">
                  <Tooltip
                    title={name === '' ? 'Base model' : name}
                    body={
                      name === ''
                        ? 'Requests served by the base weights, with no adapter applied.'
                        : inPool
                          ? `Served as model id "${
                              inventory?.model_ids?.[pooled.indexOf(name)] ?? name
                            }". In the pool and always selectable; the engine stages it into a device slot on demand.`
                          : known
                            ? 'Served requests in this window but is not in the pool of the engine now reporting.'
                            : 'Served requests in this window. This source reports no adapter inventory, so pool membership is unknown.'
                    }
                    className="tip--term"
                  >
                    {name === '' ? 'base' : name}
                  </Tooltip>
                  {absent ? (
                    <span className="adapters__flag">
                      <Pill tone="warning">not in pool</Pill>
                    </span>
                  ) : null}
                </td>
                <td className="numeric emphasis">{entry ? count(entry.summary.count) : '—'}</td>
                <td className="numeric">{entry ? count(entry.generatedTokens) : '—'}</td>
                <td className="numeric">{entry ? seconds(entry.summary.ttft.p50) : '—'}</td>
                <td className="numeric">
                  {entry && entry.summary.decodeTokensPerSecond.p50 > 0
                    ? entry.summary.decodeTokensPerSecond.p50.toFixed(0)
                    : '—'}
                </td>
                <td className="numeric">{entry ? percent(entry.summary.prefillAvoided) : '—'}</td>
                <td className="numeric">
                  {entry && entry.summary.speculative.drafted > 0
                    ? percent(entry.summary.speculative.acceptRate)
                    : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="panel__footnote">
        Rows are grouped by the resolved adapter, not by the requested model id — on the Anthropic
        route an unknown model silently falls back to the base weights. A pooled adapter with no
        rows costs disk and a directory entry, not VRAM: only the resident slots are committed.
      </p>
    </Panel>
  )
}
