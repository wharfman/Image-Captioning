import { RotateCcw, FileImage } from 'lucide-react';
import { formatBytes } from '@/lib/utils';

interface ImagePreviewProps {
  previewUrl: string;
  fileName: string;
  fileSize: number;
  onReset: () => void;
}

export default function ImagePreview({ previewUrl, fileName, fileSize, onReset }: ImagePreviewProps) {
  return (
    <div className="flex h-full flex-col gap-4">
      <div className="relative flex-1 overflow-hidden rounded-2xl border border-white/10 bg-black/30">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={previewUrl}
          alt="Uploaded preview awaiting caption"
          className="h-full w-full object-contain"
        />
      </div>
      <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3">
        <div className="flex min-w-0 items-center gap-2 text-xs text-slate-400">
          <FileImage className="h-4 w-4 shrink-0 text-slate-500" />
          <span className="truncate font-medium text-slate-300">{fileName}</span>
          <span className="shrink-0 text-slate-600">•</span>
          <span className="shrink-0">{formatBytes(fileSize)}</span>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="flex shrink-0 items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-white/20 hover:bg-white/5"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Replace
        </button>
      </div>
    </div>
  );
}
