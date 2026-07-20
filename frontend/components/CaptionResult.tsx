'use client';

import { useState } from 'react';
import { Copy, Check, FileText } from 'lucide-react';

interface CaptionResultProps {
  caption: string;
  onChange: (value: string) => void;
}

function CopyButton({ getText, label }: { getText: () => string; label: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(getText());
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard permissions denied -- silently no-op, button state doesn't flip.
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
        copied
          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
          : 'border-white/10 text-slate-300 hover:border-white/20 hover:bg-white/5'
      }`}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? 'Copied' : label}
    </button>
  );
}

export default function CaptionResult({ caption, onChange }: CaptionResultProps) {
  return (
    <div className="flex h-full flex-col gap-2 rounded-2xl border border-white/10 bg-white/[0.02] p-4 animate-slide-up">
      <div className="flex items-center justify-between">
        <label htmlFor="caption-editor" className="flex items-center gap-1.5 text-xs font-medium text-slate-400">
          <FileText className="h-3.5 w-3.5" />
          Generated caption (editable)
        </label>
        <CopyButton getText={() => caption} label="Copy caption" />
      </div>
      <textarea
        id="caption-editor"
        value={caption}
        onChange={(e) => onChange(e.target.value)}
        rows={8}
        className="flex-1 resize-none rounded-xl border border-white/10 bg-black/20 p-3 text-sm leading-relaxed text-slate-100 outline-none transition-colors focus:border-accent-500/50 focus:ring-1 focus:ring-accent-500/30"
        placeholder="Caption text will appear here..."
      />
    </div>
  );
}
