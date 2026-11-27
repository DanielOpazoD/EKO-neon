import type { AcquisitionSettings } from '@/simulator/renderer/types';

export type BmodePresetId = 'standard' | 'endocardium' | 'smooth' | 'penetration';

export type BmodePresetPatch = Pick<
  AcquisitionSettings,
  | 'dynamicRangeDb'
  | 'sri'
  | 'spatialCompound'
  | 'edgeEnhance'
  | 'persistence'
  | 'grayMap'
  | 'harmonics'
  | 'frequencyMHz'
>;

export interface BmodeImagePreset {
  id: BmodePresetId;
  label: string;
  subtitle: string;
  description: string;
  patch: BmodePresetPatch;
}

/**
 * 1-click clinical B-mode texture profiles mapped strictly to real console acquisition parameters
 * (analogous to GE Vivid / Philips EPIQ cardiac exam presets: Standard, Endocardium, Smooth/SRI,
 * and Penetration).
 */
export const BMODE_IMAGE_PRESETS: readonly BmodeImagePreset[] = [
  {
    id: 'standard',
    label: 'Estándar',
    subtitle: '70 dB · THI',
    description:
      'Perfil cardíaco equilibrado (calibrado CAMUS): rango dinámico 70 dB, armónicos 2,5 MHz y SRI moderado.',
    patch: {
      dynamicRangeDb: 70,
      sri: 0.35,
      spatialCompound: 0.4,
      edgeEnhance: 0.1,
      persistence: 0.35,
      grayMap: 'clinical',
      harmonics: true,
      frequencyMHz: 2.5,
    },
  },
  {
    id: 'endocardium',
    label: 'Endocardio',
    subtitle: '54 dB · Alto',
    description:
      'Alto contraste endocárdico para trazado de Simpson y bordes parietales: RD 54 dB, mapa alto contraste, 3,0 MHz y realce de bordes.',
    patch: {
      dynamicRangeDb: 54,
      sri: 0.45,
      spatialCompound: 0.65,
      edgeEnhance: 0.3,
      persistence: 0.3,
      grayMap: 'high-contrast',
      harmonics: true,
      frequencyMHz: 3.0,
    },
  },
  {
    id: 'smooth',
    label: 'Suave / SRI',
    subtitle: 'SRI 75%',
    description:
      'Textura suave con máxima supresión anisotrópica de speckle (SRI 0,75) y compounding alto sin perder bordes.',
    patch: {
      dynamicRangeDb: 66,
      sri: 0.75,
      spatialCompound: 0.65,
      edgeEnhance: 0.0,
      persistence: 0.5,
      grayMap: 'clinical',
      harmonics: true,
      frequencyMHz: 2.5,
    },
  },
  {
    id: 'penetration',
    label: 'Penetración',
    subtitle: '1,75 MHz',
    description:
      'Ventana acústica difícil o estructuras profundas: fundamental a 1,75 MHz, RD amplio (74 dB) y mayor persistencia.',
    patch: {
      dynamicRangeDb: 74,
      sri: 0.45,
      spatialCompound: 0.3,
      edgeEnhance: 0.1,
      persistence: 0.45,
      grayMap: 's-curve',
      harmonics: false,
      frequencyMHz: 1.75,
    },
  },
];

const approx = (a: number | undefined, b: number | undefined): boolean =>
  Math.abs((a ?? 0) - (b ?? 0)) < 1e-3;

/** Return the active B-mode preset id when current console settings match one, else null (custom). */
export function activeBmodePresetId(
  settings: Pick<
    AcquisitionSettings,
    | 'dynamicRangeDb'
    | 'sri'
    | 'spatialCompound'
    | 'edgeEnhance'
    | 'persistence'
    | 'grayMap'
    | 'harmonics'
    | 'frequencyMHz'
  >,
): BmodePresetId | null {
  for (const preset of BMODE_IMAGE_PRESETS) {
    const p = preset.patch;
    if (
      approx(settings.dynamicRangeDb, p.dynamicRangeDb) &&
      approx(settings.sri, p.sri) &&
      approx(settings.spatialCompound, p.spatialCompound) &&
      approx(settings.edgeEnhance, p.edgeEnhance) &&
      approx(settings.persistence, p.persistence) &&
      settings.grayMap === p.grayMap &&
      settings.harmonics === p.harmonics &&
      approx(settings.frequencyMHz, p.frequencyMHz)
    ) {
      return preset.id;
    }
  }
  return null;
}
