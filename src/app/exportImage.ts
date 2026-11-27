/**
 * Export the current display as PNG with a mandatory synthetic-training watermark (spec 42, 69).
 * No patient data exists in the app; the watermark makes the provenance explicit anyway.
 */
export function composeDisplay(): HTMLCanvasElement | null {
  const img = document.querySelector<HTMLCanvasElement>(
    'canvas[aria-label="Imagen ecográfica simulada"]',
  );
  const ov = document.querySelector<HTMLCanvasElement>('canvas.overlay');
  if (!img) return null;
  const out = document.createElement('canvas');
  out.width = img.width;
  out.height = img.height;
  const ctx = out.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(img, 0, 0);
  if (ov) ctx.drawImage(ov, 0, 0, out.width, out.height);
  ctx.font = 'bold 14px system-ui, sans-serif';
  ctx.fillStyle = 'rgba(255,200,87,0.9)';
  ctx.fillText('SYNTHETIC TRAINING — EchoTwin TTE — no diagnóstico', 10, out.height - 10);
  return out;
}

export function exportDisplayPng(caseId: string): void {
  const out = composeDisplay();
  if (!out) return;
  const a = document.createElement('a');
  a.download = `echotwin-${caseId}-${Date.now()}.png`;
  a.href = out.toDataURL('image/png');
  a.click();
}

/** The composed display as a PNG blob (review mode copies it to the clipboard, decision 134). */
export function displayPngBlob(): Promise<Blob | null> {
  const out = composeDisplay();
  if (!out) return Promise.resolve(null);
  return new Promise((resolve) => out.toBlob((b) => resolve(b), 'image/png'));
}

/**
 * Capture a downscaled JPEG data URL (max 640 px wide) of the composed display (ultrasound +
 * overlays + mandatory synthetic-training watermark) for the study's in-memory capture gallery and
 * printable clinical report.
 */
export function captureStudyStillDataUrl(maxWidth = 640): string | null {
  const src = composeDisplay();
  if (!src || !src.width || !src.height) return null;
  const scale = src.width > maxWidth ? maxWidth / src.width : 1;
  const w = Math.max(1, Math.round(src.width * scale));
  const h = Math.max(1, Math.round(src.height * scale));
  const out = document.createElement('canvas');
  out.width = w;
  out.height = h;
  const ctx = out.getContext('2d');
  if (!ctx) return null;
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(src, 0, 0, w, h);
  return out.toDataURL('image/jpeg', 0.86);
}

/**
 * Record a short composed WebM clip (image + overlays + ECG + synthetic training watermark)
 * directly from the display canvases and trigger download. Returns true on success.
 */
export function recordDisplayClip(caseId: string, durationMs = 3000): Promise<boolean> {
  if (typeof MediaRecorder === 'undefined') return Promise.resolve(false);
  const img = document.querySelector<HTMLCanvasElement>(
    'canvas[aria-label="Imagen ecográfica simulada"]',
  );
  const ov = document.querySelector<HTMLCanvasElement>('canvas.overlay');
  if (!img || !img.width || !img.height) return Promise.resolve(false);

  const out = document.createElement('canvas');
  out.width = img.width;
  out.height = img.height;
  const ctx = out.getContext('2d');
  if (!ctx || typeof out.captureStream !== 'function') return Promise.resolve(false);

  const mimeCandidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m));
  let recorder: MediaRecorder;
  try {
    const stream = out.captureStream(30);
    recorder = mimeType
      ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 3_500_000 })
      : new MediaRecorder(stream);
  } catch {
    return Promise.resolve(false);
  }

  return new Promise((resolve) => {
    const chunks: Blob[] = [];
    let rafId = 0;
    let stopped = false;

    const paint = () => {
      if (stopped) return;
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, out.width, out.height);
      ctx.drawImage(img, 0, 0, out.width, out.height);
      if (ov) ctx.drawImage(ov, 0, 0, out.width, out.height);
      ctx.font = 'bold 14px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,200,87,0.9)';
      ctx.fillText('SYNTHETIC TRAINING — EchoTwin TTE — no diagnóstico', 10, out.height - 10);
      rafId = requestAnimationFrame(paint);
    };

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    recorder.onerror = () => {
      stopped = true;
      cancelAnimationFrame(rafId);
      resolve(false);
    };
    recorder.onstop = () => {
      stopped = true;
      cancelAnimationFrame(rafId);
      if (!chunks.length) {
        resolve(false);
        return;
      }
      const blob = new Blob(chunks, { type: mimeType ?? 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.download = `echotwin-${caseId}-${Date.now()}.webm`;
      a.href = url;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      resolve(true);
    };

    paint();
    recorder.start(250);
    window.setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, Math.max(800, durationMs));
  });
}

