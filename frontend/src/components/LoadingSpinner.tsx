export function LoadingSpinner() {
  return (
    <div className="state-wrapper">
      <div className="loading-grid">
        {Array.from({ length: 25 }, (_, i) => (
          <div key={i} className="loading-dot" />
        ))}
      </div>
      <div className="loading-label">Simulating paths…</div>
      <div className="loading-sub">GBM · antithetic variates · 10k paths</div>
    </div>
  );
}
