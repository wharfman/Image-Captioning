export const MAX_FILE_BYTES = 5 * 1024 * 1024; // 5MB

export interface FileValidationResult {
  valid: boolean;
  error?: string;
}

export function validateImageFile(file: File): FileValidationResult {
  if (!file.type.startsWith('image/')) {
    return { valid: false, error: 'Only image files are accepted (PNG, JPG, WEBP, GIF...).' };
  }
  if (file.size > MAX_FILE_BYTES) {
    return { valid: false, error: 'That image is larger than the 5MB limit.' };
  }
  return { valid: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ');
}
