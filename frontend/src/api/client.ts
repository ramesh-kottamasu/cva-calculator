import axios from 'axios';
import type { TradeInput, ExposureResponse } from '../types/api';

export async function calculateExposure(trade: TradeInput): Promise<ExposureResponse> {
  const { data } = await axios.post<ExposureResponse>('/api/exposure', trade);
  return data;
}
