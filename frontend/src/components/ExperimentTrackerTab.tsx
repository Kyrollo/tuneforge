import React, { useState } from 'react';
import {
  FineTuningJob,
  TrainingStatus,
  TuningMethod,
} from '../types/tuneforge';
import {
  CheckCircle2,
  Clock,
  Flame,
  Cpu,
  Layers,
  Activity,
  HardDrive,
  Download,
  Sliders,
  ChevronRight,
  TrendingDown,
} from 'lucide-react';

interface ExperimentTrackerTabProps {
  jobs: FineTuningJob[];
  selectedJobId: string | null;
  onSelectJob: (id: string) => void;
}

export const ExperimentTrackerTab: React.FC<ExperimentTrackerTabProps> = ({
  jobs,
  selectedJobId,
  onSelectJob,
}) => {
  const [filterStatus, setFilterStatus] = useState<string>('ALL');

  const filteredJobs = jobs.filter((j) => {
    if (filterStatus === 'ALL') return true;
    return j.status === filterStatus;
  });

  const activeJob = jobs.find((j) => j.job_id === selectedJobId) || jobs[0];

  const getStatusBadge = (status: TrainingStatus) => {
    switch (status) {
      case 'COMPLETED':
        return {
          icon: CheckCircle2,
          color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
          label: 'Completed & Exported',
        };
      case 'TRAINING':
        return {
          icon: Flame,
          color: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
          label: 'Training in Progress',
        };
      case 'QUEUED':
        return {
          icon: Clock,
          color: 'bg-slate-700/50 text-slate-300 border-slate-600/50',
          label: 'Queued',
        };
      case 'FAILED':
        return {
          icon: Clock,
          color: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
          label: 'Failed',
        };
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Column: Experiments List */}
      <div className="lg:col-span-4 space-y-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-200">Fine-Tuning Experiments</h3>
            <span className="text-xs text-slate-500">{filteredJobs.length} jobs</span>
          </div>

          {/* Filter Pills */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {['ALL', 'COMPLETED', 'TRAINING', 'QUEUED'].map((st) => (
              <button
                key={st}
                onClick={() => setFilterStatus(st)}
                className={`px-2 py-1 text-[11px] rounded-lg font-medium transition-colors ${
                  filterStatus === st
                    ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                    : 'bg-slate-800/60 text-slate-400 hover:text-slate-200 border border-transparent'
                }`}
              >
                {st}
              </button>
            ))}
          </div>

          {/* Job Queue List */}
          <div className="space-y-2 max-h-[640px] overflow-y-auto pr-1">
            {filteredJobs.map((job) => {
              const isSelected = activeJob?.job_id === job.job_id;
              const badge = getStatusBadge(job.status);
              const BadgeIcon = badge.icon;

              return (
                <div
                  key={job.job_id}
                  onClick={() => onSelectJob(job.job_id)}
                  className={`p-3 rounded-lg cursor-pointer transition-all border text-left ${
                    isSelected
                      ? 'bg-slate-800 border-orange-500/50 shadow-md shadow-orange-500/5'
                      : 'bg-slate-900/60 hover:bg-slate-800/40 border-slate-800 text-slate-400'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-xs text-slate-300 font-medium">
                      {job.job_id}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-mono font-bold text-slate-200">
                        {job.final_eval_loss ? `Loss: ${job.final_eval_loss.toFixed(2)}` : 'Queued'}
                      </span>
                    </div>
                  </div>

                  <div className="text-xs font-medium text-slate-200 truncate mb-1">
                    {job.job_name}
                  </div>

                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-500 font-mono">
                      {job.base_model} ({job.method})
                    </span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${badge.color}`}
                    >
                      {job.status}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Right Column: Experiment Details & Loss Trajectory */}
      <div className="lg:col-span-8 space-y-4">
        {activeJob ? (
          <>
            {/* Experiment Banner */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-mono text-orange-400 font-semibold">
                      EXPERIMENT: {activeJob.job_id}
                    </span>
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded uppercase ${
                        getStatusBadge(activeJob.status).color
                      }`}
                    >
                      {activeJob.status}
                    </span>
                  </div>
                  <h2 className="text-lg font-bold text-white">{activeJob.job_name}</h2>
                  <p className="text-xs font-mono text-slate-400">
                    Base: {activeJob.base_model} | Method: {activeJob.method} ({activeJob.quantization}) | Dataset: {activeJob.dataset_name} ({activeJob.dataset_samples} samples)
                  </p>
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-center">
                    <span className="text-[10px] text-slate-500 block uppercase font-sans">Final Loss</span>
                    <span className="text-base font-bold text-emerald-400">
                      {activeJob.final_eval_loss ? activeJob.final_eval_loss.toFixed(4) : '—'}
                    </span>
                  </div>

                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-center">
                    <span className="text-[10px] text-slate-500 block uppercase font-sans">Perplexity</span>
                    <span className="text-base font-bold text-cyan-400">
                      {activeJob.perplexity ? activeJob.perplexity.toFixed(2) : '—'}
                    </span>
                  </div>

                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-center">
                    <span className="text-[10px] text-slate-500 block uppercase font-sans">Trainable %</span>
                    <span className="text-base font-bold text-orange-400">
                      {activeJob.trainable_params_pct}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Loss Curve Visualizer */}
              <div className="border border-slate-800 rounded-lg bg-slate-950/60 p-4 mb-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-orange-400" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      Optimization Loss Convergence Trajectory
                    </h3>
                  </div>
                  <div className="flex items-center gap-3 text-xs font-mono">
                    <span className="flex items-center gap-1.5 text-orange-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-orange-500" /> Train Loss
                    </span>
                    <span className="flex items-center gap-1.5 text-cyan-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" /> Val Loss
                    </span>
                  </div>
                </div>

                {activeJob.loss_history.length > 0 ? (
                  <div className="h-44 flex items-end gap-1.5 pt-4 px-2 bg-slate-950 rounded border border-slate-800/80 overflow-x-auto">
                    {activeJob.loss_history.map((step, idx) => {
                      const maxLoss = 3.5;
                      const trainHeight = Math.max(10, Math.min(100, (step.training_loss / maxLoss) * 100));
                      const valHeight = Math.max(10, Math.min(100, (step.validation_loss / maxLoss) * 100));

                      return (
                        <div
                          key={idx}
                          className="flex-1 flex flex-col justify-end items-center gap-0.5 group relative min-w-[8px]"
                          title={`Step ${step.step} | Train: ${step.training_loss.toFixed(3)} | Val: ${step.validation_loss.toFixed(3)}`}
                        >
                          <div
                            style={{ height: `${valHeight}%` }}
                            className="w-full bg-cyan-500/40 rounded-t-sm"
                          />
                          <div
                            style={{ height: `${trainHeight}%` }}
                            className="w-full bg-orange-500 rounded-t-sm group-hover:bg-orange-400 transition-colors"
                          />
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="p-8 text-center text-xs text-slate-500">
                    Job is queued in Celery scheduler. Loss trajectory will appear once worker initiates gradient descent.
                  </div>
                )}
              </div>

              {/* Hyperparameters & GGUF Export Card */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-2">
                  <h4 className="font-semibold text-slate-300 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-orange-400" />
                    Hyperparameter Configuration
                  </h4>
                  <div className="grid grid-cols-2 gap-2 font-mono text-slate-400">
                    <div>LoRA Rank (r): <span className="text-white font-bold">{activeJob.lora_rank}</span></div>
                    <div>LoRA Alpha (α): <span className="text-white font-bold">{activeJob.lora_alpha}</span></div>
                    <div>Learning Rate: <span className="text-white font-bold">{activeJob.learning_rate}</span></div>
                    <div>Epochs: <span className="text-white font-bold">{activeJob.epochs}</span></div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-2">
                  <h4 className="font-semibold text-slate-300 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                    <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                    Model Artifacts &amp; Export
                  </h4>
                  {activeJob.gguf_exported ? (
                    <div>
                      <span className="text-emerald-400 font-semibold block mb-1">
                        &bull; Quantized GGUF Checkpoint Ready
                      </span>
                      <span className="font-mono text-slate-400 text-[11px] block truncate">
                        {activeJob.export_path}
                      </span>
                    </div>
                  ) : (
                    <span className="text-slate-500">Export pending training completion</span>
                  )}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="p-12 text-center rounded-xl bg-slate-900 border border-slate-800 text-slate-400">
            No experiment selected
          </div>
        )}
      </div>
    </div>
  );
};
