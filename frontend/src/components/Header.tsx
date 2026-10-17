import React from 'react';
import { Flame, Layers, TrendingDown, Cpu, RefreshCw, HardDrive } from 'lucide-react';
import { MLRegistryKPIs } from '../types/tuneforge';

interface HeaderProps {
  kpis: MLRegistryKPIs | null;
  isLoading: boolean;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ kpis, isLoading, onRefresh }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50 px-6 py-4">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-rose-600 text-white shadow-lg shadow-orange-500/20">
            <Flame className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white font-sans">
                TuneForge
              </h1>
              <span className="px-2 py-0.5 text-xs font-semibold uppercase tracking-wider rounded-full bg-orange-500/10 text-orange-400 border border-orange-500/30">
                QLoRA Studio
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Parameter-Efficient Fine-Tuning, Loss Trajectory Tracking &amp; GGUF Model Registry
            </p>
          </div>
        </div>

        {/* Real-time ML Badges */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            <span className="text-slate-400">Completed Models:</span>
            <span className="font-semibold text-slate-200">
              {kpis ? `${kpis.completed_models} / ${kpis.total_jobs}` : '—'}
            </span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <TrendingDown className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">Perplexity Gain:</span>
            <span className="font-semibold text-emerald-400">
              {kpis ? `-${kpis.avg_perplexity_improvement}%` : '—'}
            </span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">VRAM Footprint:</span>
            <span className="font-semibold text-cyan-400">
              {kpis ? `${kpis.total_vram_allocated_gb} / 16.0 GB` : '—'}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-emerald-500/30 text-xs text-emerald-400">
            <Cpu className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Dedicated NVIDIA GPU</span>
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
          </div>

          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50"
            title="Refresh Training Experiments"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-orange-400' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
