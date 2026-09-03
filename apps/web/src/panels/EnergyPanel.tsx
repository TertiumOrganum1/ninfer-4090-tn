import { BandChart, Legend } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Panel, Pill, Stat } from '../components/ui'
import { summarizeEnergy } from '../lib/derive'
import { energyPerMillionTokens, joules, joulesPerToken, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { ThroughputRecord } from '../lib/records'
import { bands } from '../lib/series'
import type { GpuTelemetry } from '../lib/telemetry'

/**
 * What the work costs, in joules.
 *
 * Board energy comes from the GPU's own cumulative counter, so the totals here are measured rather
 * than integrated from power samples. The prefill/decode split is not: the board refreshes power
 * at roughly 50 Hz while a decode round is shorter than that, so the split is an estimate over an
 * exact total and the residual says how much of the total it fails to explain.
 *
 * Unlike the rest of the board telemetry, energy is part of the record schema, so this panel works
 * on a replayed log.
 */
export function EnergyPanel({
  records,
  gpu,
  replay,
}: {
  records: ThroughputRecord[]
  gpu: GpuTelemetry | undefined
  replay: boolean
}) {
  const energy = summarizeEnergy(records)

  if (!energy.available) {
    return (
      <Panel title="Energy" hint="energyServed" className="panel--wide" note="NVML">
        {/* Many GeForce boards do not implement nvmlDeviceGetTotalEnergyConsumption. That is a
            missing counter, not a zero, so nothing is drawn rather than a flat line at zero. */}
        <Empty>
          {records.length === 0
            ? replay
              ? 'no throughput records in the loaded log'
              : 'no samples yet'
            : gpu?.energy_available === false
              ? 'this board exposes no cumulative energy counter'
              : 'no energy in the sampled window'}
        </Empty>
      </Panel>
    )
  }

  const first = records[0]!
  const last = records[records.length - 1]!
  const domain: [number, number] = [
    first.timestamp_unix_ms - first.interval_seconds * 1000,
    last.timestamp_unix_ms,
  ]

  // The split is an estimate over an exact total. Past roughly a tenth of the total unexplained,
  // the phase figures should not be read closely, so they are marked rather than quietly shown.
  const residual = Math.abs(energy.residualFraction)
  const splitTrusted = residual <= 0.1
  const idleShare = energy.boardJoules === 0 ? 0 : energy.idleJoules / energy.boardJoules

  return (
    <Panel
      title="Energy"
      hint="energyServed"
      className="panel--wide"
      note={
        <>
          {joules(energy.boardJoules)} measured
          {splitTrusted ? null : (
            <>
              {' '}
              <Term k="energyResidual">
                <Pill tone="warning">split ±{percent(residual)}</Pill>
              </Term>
            </>
          )}
        </>
      }
    >
      <div className="stat-row">
        <Stat
          value={joulesPerToken(energy.servedJoulesPerToken)}
          unit="J/tok"
          label="served"
          hint="energyServed"
          tone="accent"
        />
        <Stat
          value={joulesPerToken(energy.activeJoulesPerToken)}
          unit="J/tok"
          label="active"
          hint="energyActive"
        />
        <Stat
          value={joulesPerToken(energy.prefillJoulesPerToken)}
          unit="J/tok"
          label="prefill"
          hint="energyPrefill"
          tone={splitTrusted ? 'neutral' : 'warning'}
        />
        <Stat
          value={joulesPerToken(energy.decodeJoulesPerToken)}
          unit="J/tok"
          label="decode"
          hint="energyDecode"
          tone={splitTrusted ? 'neutral' : 'warning'}
        />
        <Stat
          value={energy.idleWatts.toFixed(0)}
          unit="W"
          label="idle draw"
          hint="energyIdle"
          tone={idleShare > 0.5 ? 'warning' : 'neutral'}
        />
        <Stat
          value={energyPerMillionTokens(energy.servedJoulesPerToken)}
          label="per 1M tokens"
          hint="energyPerMillion"
        />
      </div>

      <BandChart
        label="Board joules per interval, by what drew them"
        domain={domain}
        series={[
          {
            name: 'prefill',
            bands: bands(records, (r) => r.energy?.prefill_joules ?? 0),
            color: CHART.blue,
          },
          {
            name: 'decode',
            bands: bands(records, (r) => r.energy?.decode_joules ?? 0),
            color: CHART.accent,
          },
          {
            name: 'idle',
            bands: bands(records, (r) => r.energy?.idle_joules ?? 0),
            color: CHART.violet,
          },
        ]}
        unit="J"
        stack
        legend={
          <Legend
            items={[
              { label: 'prefill', color: CHART.blue, hint: GLOSSARY.energyPrefill.body },
              { label: 'decode', color: CHART.accent, hint: GLOSSARY.energyDecode.body },
              { label: 'idle', color: CHART.violet, hint: GLOSSARY.energyIdle.body },
            ]}
          />
        }
        caption={
          <>
            interval bands, not snapshots <Info k="intervalBands" />
          </>
        }
      />

      <p className="panel__footnote">
        <Term k="energyServed">Served</Term> prices every joule the board drew, including the{' '}
        {percent(idleShare)} of this window it spent idle, which is what the work actually costs.{' '}
        <Term k="energyActive">Active</Term> removes the measured {energy.idleWatts.toFixed(0)} W
        baseline and tracks the schedule rather than the duty cycle.{' '}
        {splitTrusted
          ? `The phase split leaves ${percent(residual)} of measured energy unexplained.`
          : `The phase split leaves ${percent(residual)} of measured energy unexplained, so read prefill and decode as indicative only. The measured total is unaffected.`}{' '}
        Board energy excludes the CPU and the rest of the platform, so it is not a wall-socket
        figure.
        {gpu?.power_limit_watts
          ? ` Energy per token is a function of the ${gpu.power_limit_watts.toFixed(0)} W board limit, not a fixed property of the engine.`
          : null}
      </p>
      <p className="panel__footnote">
        <Term k="energyPerMillion">Per million tokens</Term> restates the same measurement in the
        denominator inference is priced in, so it can be multiplied by a local electricity rate and
        compared against a published $/1M-token figure:{' '}
        {energyPerMillionTokens(energy.servedJoulesPerToken)} served
        {energy.prefillJoulesPerToken === null
          ? null
          : `, ${energyPerMillionTokens(energy.prefillJoulesPerToken)} per million prefilled`}
        {energy.decodeJoulesPerToken === null
          ? null
          : `, ${energyPerMillionTokens(energy.decodeJoulesPerToken)} per million decoded`}
        . It carries no information the per-token figures do not; energy stays per token above
        because it composes additively across phases and this restatement does not.
      </p>
    </Panel>
  )
}
