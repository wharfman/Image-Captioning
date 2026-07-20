import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'Image Captioner',
  description: 'Generate accurate, accessible image captions with a CLIP + GRU captioning model.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body className="min-h-screen bg-surface-950 font-sans text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
