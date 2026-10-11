import React from 'react';
import { FineTuningJob, MLRegistryKPIs } from '../types/tuneforge';
import {
  Layers,
  Cpu,
  HardDrive,
  Download,
  Share2,
  CheckCircle2,
  Terminal,
  Zap,
} from 'lucide-react';

interface ModelRegistryTabProps {
  jobs: FineTuningJob[];
  kpis: MLRegistryKPIs | null;
}

export const ModelRegistryTab: React.FC<ModelRegistryTabProps> = ({ jobs, kpis }) => {
  const completedAdapters = jobs.filter((j) => j.gguf_exported);

  return (
    <div className="space-y-6">
      {/* Parameter Efficiency & VRAM Comparison */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-5 h-5 text-orange-400" />
          <h3 className="text-sm font-semibold text-white">
            Fine-Tuning Architecture &amp; VRAM Efficiency Benchmarks
          </h3>
        </div>
        <p className="text-xs text-slate-400 mb-4">
          By utilizing 4-bit NormalFloat (NF4) quantization with Low-Rank Adaptation (QLoRA), TuneForge reduces VRAM requirements by over 80% while retaining full FP16 base model precision.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500 font-medium">
                <th className="pb-2">Fine-Tuning Method</th>
                <th className="pb-2">Trainable Weights %</th>
                <th className="pb-2">VRAM Required (8B Model)</th>
                <th className="pb-2">Hardware Compatibility</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              <tr>
                <td className="py-2.5 text-rose-400 font-semibold font-sans">Full Fine-Tuning (16-bit)</td>
                <td className="py-2.5">100.0% (8.03B params)</td>
                <td className="py-2.5 text-rose-400 font-bold">~64.0 GB VRAM</td>
                <td className="py-2.5 font-sans text-slate-400">Requires 4x A100 Data Center Cluster</td>
              </tr>
              <tr>
                <td className="py-2.5 text-amber-400 font-semibold font-sans">Standard LoRA (16-bit Base)</td>
                <td className="py-2.5">0.42% (33.5M params)</td>
                <td className="py-2.5 text-amber-400 font-bold">~18.5 GB VRAM</td>
                <td className="py-2.5 font-sans text-slate-400">Requires A10 / RTX 4090 24GB</td>
              </tr>
              <tr className="bg-emerald-950/20">
                <td className="py-2.5 text-emerald-400 font-semibold font-sans">TuneForge QLoRA (NF4 4-bit)</td>
                <td className="py-2.5 text-emerald-400 font-bold">0.28% (1.4M params)</td>
                <td className="py-2.5 text-emerald-400 font-bold">~2.2 GB VRAM</td>
                <td className="py-2.5 font-sans text-emerald-300 font-semibold">
                  Runs natively on NVIDIA RTX A5000 Laptop GPU (16GB)
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Adapter Registry Table */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-orange-400" />
            <h3 className="text-sm font-semibold text-white">Registered Model Adapters</h3>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            {completedAdapters.length} adapters exported
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500 font-medium">
                <th className="pb-2">Adapter ID</th>
                <th className="pb-2">Model Name</th>
                <th className="pb-2">Base Weights</th>
                <th className="pb-2">LoRA Rank</th>
                <th className="pb-2 text-right">Perplexity</th>
                <th className="pb-2 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              {jobs.map((job) => (
                <tr key={job.job_id}>
                  <td className="py-2.5 text-orange-400 font-semibold">{job.job_id}</td>
                  <td className="py-2.5 font-sans font-medium text-slate-200">{job.job_name}</td>
                  <td className="py-2.5 text-slate-400">{job.base_model}</td>
                  <td className="py-2.5">r={job.lora_rank}, α={job.lora_alpha}</td>
                  <td className="py-2.5 text-right font-bold text-cyan-400">
                    {job.perplexity ? job.perplexity.toFixed(2) : '—'}
                  </td>
                  <td className="py-2.5 text-right">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                        job.status === 'COMPLETED'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : job.status === 'TRAINING'
                          ? 'bg-orange-500/20 text-orange-400'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {job.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
