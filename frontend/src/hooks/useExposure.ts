import { useState, useCallback } from 'react';
import type { TradeInput, ExposureResponse } from '../types/api';
import { calculateExposure } from '../api/client';

interface UseExposureResult {
  data: ExposureResponse | null;
  loading: boolean;
  error: string | null;
  calculate: (trade: TradeInput) => Promise<void>;
}

export function useExposure(): UseExposureResult {
  const [data, setData] = useState<ExposureResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calculate = useCallback(async (trade: TradeInput) => {
    setLoading(true);
    setError(null);
    try {
      const result = await calculateExposure(trade);
      setData(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Calculation failed';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, calculate };
}
