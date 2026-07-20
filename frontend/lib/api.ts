const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export class CaptionApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'CaptionApiError';
    this.status = status;
  }
}

export async function requestCaption(file: File, signal?: AbortSignal): Promise<string> {
  const formData = new FormData();
  formData.append('image', file);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/caption`, {
      method: 'POST',
      body: formData,
      signal,
    });
  } catch {
    throw new CaptionApiError(
      `Could not reach the captioning API at ${API_URL}. Is the FastAPI server running?`
    );
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response body wasn't JSON -- fall back to the generic message
    }
    throw new CaptionApiError(detail, response.status);
  }

  const data = await response.json();
  return data.caption as string;
}
