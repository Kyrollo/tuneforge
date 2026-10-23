import {
  CreateJobRequest,
  FineTuningJob,
  InferenceCompareRequest,
  MLRegistryKPIs,
} from '../types/tuneforge';

const BASE_URL = '/api/v1';

export async function fetchJobs(): Promise<FineTuningJob[]> {
  const res = await fetch(`${BASE_URL}/jobs`);
  if (!res.ok) {
    throw new Error(`Failed to fetch fine-tuning jobs: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchJob(id: string): Promise<FineTuningJob> {
  const res = await fetch(`${BASE_URL}/jobs/${id}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch job ${id}: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchStats(): Promise<MLRegistryKPIs> {
  const res = await fetch(`${BASE_URL}/stats`);
  if (!res.ok) {
    throw new Error(`Failed to fetch ML registry KPIs: ${res.statusText}`);
  }
  return res.json();
}

export async function createJob(payload: CreateJobRequest): Promise<FineTuningJob> {
  const res = await fetch(`${BASE_URL}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    const message = errorBody?.detail?.detail || errorBody?.detail || res.statusText;
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return res.json();
}

export async function compareInference(payload: InferenceCompareRequest): Promise<any> {
  const res = await fetch(`${BASE_URL}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    const message = errorBody?.detail?.detail || errorBody?.detail || res.statusText;
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return res.json();
}
