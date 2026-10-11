export type TuningMethod = 'QLORA' | 'LORA' | 'FULL_FINETUNE' | 'DPO';

export type QuantizationBit = 'INT4' | 'INT8' | 'FP16';

export type TrainingStatus = 'QUEUED' | 'TRAINING' | 'COMPLETED' | 'FAILED';

export interface LossStep {
  step: number;
  epoch: number;
  training_loss: number;
  validation_loss: number;
  learning_rate: number;
  gpu_vram_gb: number;
}

export interface FineTuningJob {
  job_id: string;
  job_name: string;
  base_model: string;
  dataset_name: string;
  dataset_samples: number;
  method: TuningMethod;
  quantization: QuantizationBit;
  lora_rank: number;
  lora_alpha: number;
  learning_rate: number;
  epochs: number;
  status: TrainingStatus;
  current_step: number;
  total_steps: number;
  final_eval_loss?: number | null;
  perplexity?: number | null;
  trainable_params_pct: number;
  gguf_exported: boolean;
  export_path?: string | null;
  loss_history: LossStep[];
  created_at: string;
}

export interface MLRegistryKPIs {
  total_jobs: number;
  completed_models: number;
  active_training_jobs: number;
  avg_perplexity_improvement: number;
  total_vram_allocated_gb: number;
  registered_adapters: string[];
}

export interface CreateJobRequest {
  job_name: string;
  base_model?: string;
  dataset_text: string;
  method?: TuningMethod;
  lora_rank?: number;
  learning_rate?: number;
  epochs?: number;
}

export interface InferenceCompareRequest {
  prompt: string;
  model?: string;
  adapter_id?: string;
  temperature?: number;
  max_tokens?: number;
}
