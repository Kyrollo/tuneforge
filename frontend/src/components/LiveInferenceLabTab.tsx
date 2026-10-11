import React, { useState } from 'react';
import { compareInference } from '../lib/api';
import {
  Play,
  Cpu,
  Layers,
  Sparkles,
  AlertTriangle,
  Clock,
  Zap,
  Sliders,
  CheckCircle2,
} from 'lucide-react';

const PRESET_PROMPTS = [
  {
    title: 'SRE Post-Mortem Incident RCA',
    prompt:
      'Incident Post-Mortem: Production Kubernetes cluster pods in namespace billing failed with CrashLoopBackOff. CoreDNS latency spiked to 2.4s. Detail immediate remediation and permanent architecture fixes.',
  },
  {
    title: 'PostgreSQL Query Optimization',
    prompt:
      'Explain how to optimize this slow query: SELECT * FROM audit_logs WHERE metadata->>\'status\' = \'FAILED\' ORDER BY created_at DESC LIMIT 100 on a 50M row table.',
  },
  {
    title: 'HIPAA Medical Record Discharge',
    prompt:
      'Draft a clinical summary for a 54yo patient admitted with acute bacterial pneumonia, successfully treated with IV ceftriaxone, discharged with oral azithromycin.',
  },
];

export const LiveInferenceLabTab: React.FC = () => {
  const [prompt, setPrompt] = useState<string>(PRESET_PROMPTS[0].prompt);
  const [model, setModel] = useState<string>('qwen2.5:0.5b');
  const [adapterId, setAdapterId] = useState<string>('ft-legal-801');
  const [temperature, setTemperature] = useState<number>(0.2);
  const [maxTokens, setMaxTokens] = useState<number>(200);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);

  const handleRunComparison = async () => {
    if (!prompt.trim()) return;
    setIsRunning(true);
    setError(null);

    try {
      const data = await compareInference({
        prompt: prompt.trim(),
        model,
        adapter_id: adapterId,
        temperature,
        max_tokens: maxTokens,
      });
      setResult(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to execute comparison inference on local GPU');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Column: Prompt Input & Parameters */}
      <div className="lg:col-span-5 space-y-4">
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-orange-400" />
              <h2 className="text-base font-bold text-white font-sans">
                Fine-Tuning Inference Lab
              </h2>
            </div>
            <span className="px-2 py-0.5 text-xs font-mono rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Ollama RTX A5000
            </span>
          </div>

          {/* Presets */}
          <div className="mb-4">
            <span className="text-xs text-slate-400 block mb-2 font-medium">
              Domain Evaluation Scenarios:
            </span>
            <div className="flex flex-wrap gap-2">
              {PRESET_PROMPTS.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => setPrompt(preset.prompt)}
                  className="px-2.5 py-1 text-xs rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-700/60"
                >
                  {preset.title}
                </button>
              ))}
            </div>
          </div>

          {/* Model & Adapter Selectors */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className="text-xs text-slate-400 block mb-1 font-medium">
                Base Model:
              </label>
              <input
                type="text"
                disabled
                value={model}
                className="w-full p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-400"
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 block mb-1 font-medium">
                LoRA Adapter:
              </label>
              <select
                value={adapterId}
                onChange={(e) => setAdapterId(e.target.value)}
                className="w-full p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none focus:border-orange-500/50"
              >
                <option value="ft-legal-801">ft-legal-801 (SRE Post-Mortem)</option>
                <option value="ft-code-802">ft-code-802 (SQL Optimizer)</option>
              </select>
            </div>
          </div>

          {/* Prompt Textarea */}
          <div className="mb-4">
            <label className="text-xs text-slate-400 block mb-1.5 font-medium">
              Evaluation Prompt:
            </label>
            <textarea
              rows={6}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none focus:border-orange-500/50 resize-none"
              placeholder="Enter evaluation prompt..."
            />
          </div>

          {/* Action Button */}
          <button
            onClick={handleRunComparison}
            disabled={isRunning || !prompt.trim()}
            className="w-full py-3 px-4 rounded-xl font-semibold text-xs tracking-wide uppercase bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-400 hover:to-rose-500 text-white shadow-lg shadow-orange-500/20 flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <Cpu className="w-4 h-4 animate-spin text-white" />
                <span>Executing Comparison on Workstation GPU...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 text-white fill-current" />
                <span>Run Comparison Inference</span>
              </>
            )}
          </button>

          {error && (
            <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-start gap-2 text-xs text-rose-300">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Right Column: Side-by-Side Outputs */}
      <div className="lg:col-span-7 space-y-4">
        {result ? (
          <div className="space-y-4">
            {/* Telemetry Header */}
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-[11px] text-slate-400 block mb-0.5">Inference Latency</span>
                <div className="flex items-center gap-1.5 font-mono font-bold text-white text-base">
                  <Clock className="w-3.5 h-3.5 text-orange-400" />
                  <span>{result.latency_ms}ms</span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-[11px] text-slate-400 block mb-0.5">Tokens Generated</span>
                <div className="flex items-center gap-1.5 font-mono font-bold text-cyan-400 text-base">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{result.tokens_generated}</span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-[11px] text-slate-400 block mb-0.5">Throughput</span>
                <div className="flex items-center gap-1.5 font-mono font-bold text-emerald-400 text-base">
                  <Zap className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{result.tokens_per_sec} tps</span>
                </div>
              </div>
            </div>

            {/* Fine-Tuned Output Card */}
            <div className="p-5 rounded-xl bg-slate-900 border border-orange-500/30">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-orange-400" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-orange-400">
                    Fine-Tuned Adapter Output ({result.adapter_id})
                  </h3>
                </div>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-400">
                  Domain Aligned
                </span>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 leading-relaxed font-mono whitespace-pre-wrap max-h-72 overflow-y-auto">
                {result.fine_tuned_output}
              </div>
            </div>

            {/* Baseline Output Card */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-slate-400" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Base Model Baseline ({result.base_model})
                </h3>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-400 leading-relaxed font-mono">
                {result.baseline_output}
              </div>
            </div>
          </div>
        ) : (
          <div className="p-12 text-center rounded-xl bg-slate-900 border border-slate-800 text-slate-500 space-y-3">
            <Cpu className="w-10 h-10 text-slate-700 mx-auto" />
            <p className="text-xs">
              Execute comparison inference to evaluate response quality improvements between base weights and fine-tuned LoRA adapters on the local GPU.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
