'use client';

import { useCallback, useRef, useState } from 'react';
import { UploadCloud, ImageOff } from 'lucide-react';
import { validateImageFile } from '@/lib/utils';

interface DropzoneProps {
  onFileAccepted: (file: File) => void;
  disabled?: boolean;
}

export default function Dropzone({ onFileAccepted, disabled }: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      const result = validateImageFile(file);
      if (!result.valid) {
        setError(result.error ?? 'That file is not supported.');
        return;
      }
      setError(null);
      onFileAccepted(file);
    },
    [onFileAccepted]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      handleFiles(e.dataTransfer.files);
    },
    [disabled, handleFiles]
  );

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image to caption"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={`group relative flex w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-6 py-16 text-center transition-all duration-200 ${
          isDragging
            ? 'border-accent-500 bg-accent-500/10 shadow-glow'
            : 'border-white/15 bg-white/[0.02] hover:border-accent-500/50 hover:bg-white/[0.04]'
        } ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div
          className={`flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-accent-500/20 to-violet-500/20 transition-transform duration-200 ${
            isDragging ? 'scale-110' : 'group-hover:scale-105'
          }`}
        >
          <UploadCloud className="h-7 w-7 text-accent-400" />
        </div>
        <div>
          <p className="text-sm font-medium text-slate-200">
            <span className="text-accent-400">Click to upload</span> or drag and drop
          </p>
          <p className="mt-1 text-xs text-slate-500">PNG, JPG, or WEBP — up to 5MB</p>
        </div>
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300 animate-fade-in">
          <ImageOff className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
