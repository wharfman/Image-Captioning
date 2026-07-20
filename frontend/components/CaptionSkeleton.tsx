import { Loader2 } from 'lucide-react';

export default function CaptionSkeleton() {
  return (
    <div className="flex h-full flex-col gap-4 animate-fade-in">
      <div className="flex items-center gap-2 text-sm text-accent-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        Generating caption...
      </div>
      <div className="space-y-3">
        <div className="bg-shimmer h-4 w-full animate-shimmer rounded-md bg-white/5" />
        <div className="bg-shimmer h-4 w-[85%] animate-shimmer rounded-md bg-white/5" />
        <div className="bg-shimmer h-4 w-[70%] animate-shimmer rounded-md bg-white/5" />
      </div>
      <div className="mt-2 space-y-3 rounded-xl border border-white/10 bg-white/[0.02] p-4">
        <div className="bg-shimmer h-3 w-1/3 animate-shimmer rounded bg-white/5" />
        <div className="bg-shimmer h-3 w-full animate-shimmer rounded bg-white/5" />
        <div className="bg-shimmer h-3 w-2/3 animate-shimmer rounded bg-white/5" />
      </div>
    </div>
  );
}
