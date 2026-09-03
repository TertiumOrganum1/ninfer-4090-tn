// Display formatting. Every reading in the dashboard is a number a maintainer will compare
// against a log line or a metrics scrape, so these keep magnitudes explicit and never round a
// value into a different order.

export function bytes(value: number, digits = 1): string {
  if (!Number.isFinite(value) || value <= 0) return '0'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let scaled = value
  let unit = 0
  while (scaled >= 1000 && unit < units.length - 1) {
    scaled /= 1000
    unit += 1
  }
  return `${scaled.toFixed(unit === 0 ? 0 : digits)} ${units[unit]}`
}

export function count(value: number): string {
  if (!Number.isFinite(value)) return '0'
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (Math.abs(value) >= 1e4) return `${(value / 1e3).toFixed(1)}k`
  return Math.round(value).toLocaleString('en-US')
}

export function percent(fraction: number, digits = 0): string {
  if (!Number.isFinite(fraction)) return '0%'
  return `${(fraction * 100).toFixed(digits)}%`
}

export function seconds(value: number): string {
  if (!Number.isFinite(value)) return '0s'
  if (value >= 60) {
    const minutes = Math.floor(value / 60)
    return `${minutes}m ${Math.round(value - minutes * 60)}s`
  }
  if (value >= 10) return `${value.toFixed(1)}s`
  if (value >= 0.01) return `${value.toFixed(2)}s`
  return `${(value * 1000).toFixed(0)}ms`
}

export function duration(value: number): string {
  if (!Number.isFinite(value) || value < 0) return '0s'
  const hours = Math.floor(value / 3600)
  const minutes = Math.floor((value % 3600) / 60)
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m ${Math.floor(value % 60)}s`
  return `${Math.floor(value)}s`
}

export function rate(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0'
  if (value >= 1000) return Math.round(value).toLocaleString('en-US')
  return value.toFixed(value >= 100 ? 0 : 1)
}

export function clock(unixMs: number): string {
  return new Date(unixMs).toLocaleTimeString('en-US', { hour12: false })
}

/**
 * Energy, scaled so a long-running window and a single request read at the same precision.
 *
 * A watt-second is a joule, so there is only one unit here; kJ and MJ are the same unit scaled.
 */
export function joules(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 J'
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} MJ`
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)} kJ`
  return `${value.toFixed(1)} J`
}

/**
 * Joules per token. Null renders as an em dash: no tokens of that kind is not zero energy each.
 *
 * Energy per token rather than tokens per joule because energy composes additively across phases
 * while a rate does not — prefill and decode figures can be combined against their own token
 * counts, whereas averaging tokens-per-joule arithmetically is simply wrong.
 */
export function joulesPerToken(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0) return '—'
  if (value >= 100) return value.toFixed(0)
  if (value >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

/** Joules per token expressed per million tokens, in watt-hours. Exact: 1 Wh = 3600 J. */
export const WH_PER_MILLION_TOKENS_PER_JOULE = 1e6 / 3600

/**
 * Energy per million tokens — the denominator inference is priced in.
 *
 * The same measurement as {@link joulesPerToken}, rescaled. It exists because a watt-hour is the
 * unit electricity is billed in and a million tokens is the unit inference is sold in, so one
 * multiplication by a local price gives a figure directly comparable to a published $/1M-token
 * rate. Nothing here is derivable from this that is not derivable from J/token; the value is the
 * shared denominator, not new information.
 *
 * Scales Wh to kWh so a cheap prefill token and an expensive decode token both read at the same
 * precision: on this target they span roughly 49 Wh to 1.06 kWh per million.
 */
export function energyPerMillionTokens(joulesPerTok: number | null): string {
  if (joulesPerTok === null || !Number.isFinite(joulesPerTok) || joulesPerTok < 0) return '—'
  const wh = joulesPerTok * WH_PER_MILLION_TOKENS_PER_JOULE
  if (wh >= 1000) return `${(wh / 1000).toFixed(2)} kWh`
  if (wh >= 10) return `${wh.toFixed(0)} Wh`
  if (wh > 0) return `${wh.toFixed(1)} Wh`
  return '0 Wh'
}
