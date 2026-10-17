import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { ExperimentTrackerTab } from './components/ExperimentTrackerTab';
import { ModelRegistryTab } from './components/ModelRegistryTab';
import { LiveInferenceLabTab } from './components/LiveInferenceLabTab';
import { fetchJobs, fetchStats } from './lib/api';
import { FineTuningJob, MLRegistryKPIs } from './types/tuneforge';
import { FileUploadZone } from './components/FileUploadZone';
import { GitCommit, Flame, Layers, PlayCircle, FolderUp } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'experiments' | 'registry' | 'lab' | 'uploads'>('experiments');
  const [jobs, setJobs] = useState<FineTuningJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [kpis, setKpis] = useState<MLRegistryKPIs | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [fetchedJobs, fetchedKPIs] = await Promise.all([
        fetchJobs(),
        fetchStats(),
      ]);
      setJobs(fetchedJobs);
      setKpis(fetchedKPIs);
      if (fetchedJobs.length > 0 && !selectedJobId) {
        setSelectedJobId(fetchedJobs[0].job_id);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to load fine-tuning experiments from TuneForge backend');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-orange-500/30">
      {/* TuneForge Header */}
      <Header kpis={kpis} isLoading={isLoading} onRefresh={loadData} />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-800 gap-2">
          <button
            onClick={() => setActiveTab('experiments')}
            className={`flex items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition-all ${
              activeTab === 'experiments'
                ? 'border-orange-500 text-orange-400 bg-orange-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Flame className="w-4 h-4" />
            Experiment Tracker ({jobs.length})
          </button>

          <button
            onClick={() => setActiveTab('registry')}
            className={`flex items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition-all ${
              activeTab === 'registry'
                ? 'border-orange-500 text-orange-400 bg-orange-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-4 h-4" />
            Model Registry &amp; VRAM
          </button>

          <button
            onClick={() => setActiveTab('lab')}
            className={`flex items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition-all ${
              activeTab === 'lab'
                ? 'border-orange-500 text-orange-400 bg-orange-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <PlayCircle className="w-4 h-4" />
            Live Inference Lab
          </button>

          <button
            onClick={() => setActiveTab('uploads')}
            className={`flex items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-wider border-b-2 transition-all ${
              activeTab === 'uploads'
                ? 'border-orange-500 text-orange-400 bg-orange-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <FolderUp className="w-4 h-4" />
            Dataset Catalog Ingestion
          </button>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
            {error}
          </div>
        )}

        {/* Tab Views */}
        {activeTab === 'experiments' && (
          <ExperimentTrackerTab
            jobs={jobs}
            selectedJobId={selectedJobId}
            onSelectJob={setSelectedJobId}
          />
        )}

        {activeTab === 'registry' && <ModelRegistryTab jobs={jobs} kpis={kpis} />}

        {activeTab === 'lab' && <LiveInferenceLabTab />}

        {activeTab === 'uploads' && (
          <FileUploadZone
            title="Training Dataset & Corpus Ingestion"
            acceptedTypes=".jsonl,.json,.csv,.txt,.parquet"
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 px-6 py-4 text-center text-xs text-slate-600 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-2 font-mono">
          <GitCommit className="w-3.5 h-3.5 text-orange-500" />
          <span>TuneForge Production v1.0.0</span>
          <span>•</span>
          <span>QLoRA NF4 Quantized Training Architecture</span>
        </div>
        <div>
          <span>Local Engine: Dedicated NVIDIA GPU (16GB VRAM)</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
