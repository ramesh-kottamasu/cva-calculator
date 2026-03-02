import type { ExposureResponse } from '../types/api';

interface Props { data: ExposureResponse }

/** Format a signed delta value compactly. */
function fmtDelta(v: number): string {
  const sign  = v >= 0 ? '+' : '\u2212';   // '−' (minus sign, not hyphen)
  const abs   = Math.abs(v);
  if (abs >= 10_000) return `${sign}${(abs / 1000).toFixed(1)}k`;
  if (abs >= 100)    return `${sign}${abs.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  if (abs >= 1)      return `${sign}${abs.toFixed(1)}`;
  return `${sign}${abs.toFixed(2)}`;
}

export function SensitivitiesPanel({ data }: Props) {
  const sens = data.sensitivities ?? [];
  if (sens.length === 0) return null;

  return (
    <div className="sens-panel">
      <div className="sens-header">
        <span className="sens-title">CVA Sensitivities</span>
        <span className="sens-method">bump-and-reprice · seed fixed</span>
      </div>

      <table className="sens-table">
        <thead>
          <tr>
            <th className="col-label">Sensitivity</th>
            <th className="col-bump">Bump</th>
            <th className="col-num">&Delta; CVA</th>
            <th className="col-num">&Delta; DVA</th>
            <th className="col-num">&Delta; BCVA</th>
          </tr>
        </thead>
        <tbody>
          {sens.map((s) => (
            <tr key={s.label}>
              <td className="sens-label-cell">{s.label}</td>
              <td className="sens-bump-cell">{s.bump}</td>
              <td className={`sens-val ${s.delta_cva  > 0 ? 'val-cost' : s.delta_cva  < 0 ? 'val-benefit' : ''}`}>
                {fmtDelta(s.delta_cva)}
              </td>
              <td className={`sens-val ${s.delta_dva  > 0 ? 'val-benefit' : s.delta_dva  < 0 ? 'val-cost' : ''}`}>
                {fmtDelta(s.delta_dva)}
              </td>
              <td className={`sens-val ${s.delta_bcva > 0 ? 'val-cost' : s.delta_bcva < 0 ? 'val-benefit' : ''}`}>
                {fmtDelta(s.delta_bcva)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="sens-note">
        CS01: no re-simulation &nbsp;&middot;&nbsp;
        {data.product === 'fx_forward'
          ? 'Spot \u0394 / Vega: re-simulate +1% spot / +1 vol pt'
          : 'IR \u0394 / Rate Vega: re-simulate +1bp r\u2080 / +1bp \u03c3\u1d63'}
      </p>
    </div>
  );
}
