'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import Header from '@/components/Header';
import Dropzone from '@/components/Dropzone';
import ImagePreview from '@/components/ImagePreview';
import CaptionSkeleton from '@/components/CaptionSkeleton';
import CaptionResult from '@/components/CaptionResult';
import { requestCaption, CaptionApiError } from '@/lib/api';

type Status = 'idle' | 'loading' | 'success' | 'error';

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [caption, setCaption] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Revoke the object URL whenever it's replaced or the component unmounts,
  // otherwise each upload leaks a blob reference for the session's lifetime.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const runCaptioning = useCallback(async (selected: File) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus('loading');
    setErrorMessage(null);
    try {
      const result = await requestCaption(selected, controller.signal);
      setCaption(result);
      setStatus('success');
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      const message = err instanceof CaptionApiError ? err.message : 'Something went wrong generating the caption.';
      setErrorMessage(message);
      setStatus('error');
    }
  }, []);

  const handleFileAccepted = useCallback(
    (selected: File) => {
      setFile(selected);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(selected);
      });
      setCaption('');
      void runCaptioning(selected);
    },
    [runCaptioning]
  );

  const handleReset = useCallback(() => {
    abortRef.current?.abort();
    setFile(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setStatus('idle');
    setCaption('');
    setErrorMessage(null);
  }, []);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-10 px-6 py-14 sm:py-20">
      <Header />

      {!file || !previewUrl ? (
        <div className="mx-auto w-full max-w-xl animate-fade-in">
          <Dropzone onFileAccepted={handleFileAccepted} />
        </div>
      ) : (
        <section className="grid w-full grid-cols-1 gap-6 lg:grid-cols-2 lg:gap-8">
          <ImagePreview
            previewUrl={previewUrl}
            fileName={file.name}
            fileSize={file.size}
            onReset={handleReset}
          />

          <div className="flex flex-col">
            {status === 'loading' && <CaptionSkeleton />}

            {status === 'error' && (
              <div className="flex h-full flex-col items-center justify-center gap-4 rounded-2xl border border-red-500/20 bg-red-500/5 p-8 text-center animate-fade-in">
                <AlertTriangle className="h-8 w-8 text-red-400" />
                <p className="text-sm text-red-200">{errorMessage}</p>
                <button
                  type="button"
                  onClick={() => file && void runCaptioning(file)}
                  className="flex items-center gap-1.5 rounded-lg border border-red-500/30 px-3 py-1.5 text-xs font-medium text-red-200 transition-colors hover:bg-red-500/10"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Try again
                </button>
              </div>
            )}

            {status === 'success' && (
              <CaptionResult caption={caption} onChange={setCaption} />
            )}
          </div>
        </section>
      )}
    </main>
  );
}
