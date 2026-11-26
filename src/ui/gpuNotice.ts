const TIER_LABEL: Record<string, string> = { low: 'baja', medium: 'media', high: 'alta' };

/** The stats of an output the HUD's GPU notice reads (`SimOutput.stats`). */
export interface GpuNoticeStats {
  gpu?: number | string | null;
  tier?: number | string | null;
  gpuLive?: number | string | null;
}

/**
 * The HUD's GPU notice over the frame shown, from its output's stats, which speak of that frame, frozen too, and of the
 * GPU port now (`gpuLive`). Formed by the CPU tracer for want of the GPU (no WebGL2, the context lost), slower and, on
 * «auto», coarser, it says so with the frame's quality; its tip says why, and where the port stands now: «GPU de vuelta»,
 * the attempt next, or given up. The frame keeps only why: with the port's state of its capture, the tip over a frozen
 * frame said «trazador CPU hasta que vuelva la GPU (intento 1 de 5)» once the GPU was back, or five attempts later.
 * Frozen over a frame the GPU formed while the port is lost now, it says only the port's state: nothing forms a frame
 * until the unfreeze. Null when there is nothing to say.
 */
export function gpuNotice(s: GpuNoticeStats): { text: string; tip?: string } | null {
  const port = typeof s.gpuLive === 'string' ? s.gpuLive : null;
  if (typeof s.gpu === 'string' && s.gpu !== 'ok') {
    const quality = typeof s.tier === 'string' ? `, calidad ${TIER_LABEL[s.tier] ?? s.tier}` : '';
    // without a port at all its state is the very reason (no WebGL2): nothing to add, nothing to wait for
    const tip =
      port === null || port === s.gpu
        ? s.gpu
        : `${s.gpu}: cuadro formado por el trazador CPU · ${port === 'ok' ? 'GPU de vuelta' : port}`;
    return { text: `Sin GPU · trazador CPU${quality}`, tip };
  }
  return port !== null && port !== 'ok' ? { text: port } : null;
}
