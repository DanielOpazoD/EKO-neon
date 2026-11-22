import { computeHeartPose, type HeartModel, type HeartPose } from '@/simulator/anatomy/heartModel';
import { cycleStateAt, type BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import type { AssetFetcher } from '@/simulator/anatomy/roderoAsset';
import type { LabelVolume, VolumeHeart } from '@/simulator/anatomy/volumeHeart';
import type { HeartVolumeGpu } from '@/simulator/renderer/types';
import type * as RoderoHeartModule from '@/simulator/core/roderoHeart';
import type * as ColeImageModule from '@/simulator/core/coleImage';
import type { AnatomyChoice, ImageFormationChoice } from './protocol';

export interface BoundRoderoHeart {
  heart: HeartModel;
  vh: VolumeHeart;
  gpu: HeartVolumeGpu;
  endDiastole: HeartPose;
}

/**
 * Lazy on-demand loader and cache for the experimental Rodero volume anatomy (decision 352)
 * and the experimental Cole convolution image formation (decision 354).
 */
export class ExperimentalLoaders {
  private roderoLib: typeof RoderoHeartModule | null = null;
  private roderoVolume: LabelVolume | null = null;
  private roderoLoad: Promise<void> | null = null;
  private roderoFailure: string | null = null;
  private roderoBound: BoundRoderoHeart | null = null;

  private coleLib: typeof ColeImageModule | null = null;
  private coleLoad: Promise<void> | null = null;
  private coleFailure: string | null = null;

  constructor(
    private readonly fetchAsset: AssetFetcher,
    private readonly onLoaded: () => void,
  ) {}

  /**
   * The Rodero volume heart bound to the current heart model, when the experimental anatomy is chosen and its volume has
   * arrived; starts loading it the first time it is asked for. Null otherwise (the analytic heart draws meanwhile).
   */
  roderoHeart(
    anatomy: AnatomyChoice | undefined,
    heart: HeartModel,
    tables: BeatTables,
  ): BoundRoderoHeart | null {
    if (anatomy !== 'rodero') return null;
    if (!this.roderoLoad) {
      this.roderoLoad = import('@/simulator/core/roderoHeart')
        .then(async (lib) => {
          this.roderoVolume = await lib.loadRoderoVolume(this.fetchAsset);
          this.roderoLib = lib;
          this.onLoaded();
        })
        .catch((e: unknown) => {
          this.roderoFailure = e instanceof Error ? e.message : String(e);
        });
    }
    if (!this.roderoLib || !this.roderoVolume) return null;
    if (this.roderoBound?.heart !== heart) {
      const endDiastole = computeHeartPose(heart, cycleStateAt(tables, 0));
      const vh = this.roderoLib.volumeHeart(this.roderoVolume, heart, endDiastole);
      this.roderoBound = {
        heart,
        vh,
        gpu: this.roderoLib.volumeHeartGpu(vh, endDiastole),
        endDiastole,
      };
    }
    return this.roderoBound;
  }

  /**
   * The convolution image formation, when the experimental image is chosen and its chunk has arrived; starts loading it
   * the first time it is asked for. Null otherwise (the lattice image forms meanwhile).
   */
  coleImage(imageFormation: ImageFormationChoice | undefined): typeof ColeImageModule | null {
    if (imageFormation !== 'cole') return null;
    if (!this.coleLoad) {
      this.coleLoad = import('@/simulator/core/coleImage')
        .then((lib) => {
          this.coleLib = lib;
          this.onLoaded();
        })
        .catch((e: unknown) => {
          this.coleFailure = e instanceof Error ? e.message : String(e);
        });
    }
    return this.coleLib;
  }

  /** Resolves when the experimental image formation has arrived or failed (tests and tools). */
  async coleReady(imageFormation: ImageFormationChoice | undefined): Promise<void> {
    this.coleImage(imageFormation);
    await this.coleLoad;
  }

  /** Resolves when the experimental anatomy's volume has arrived or failed (tests and tools). */
  async roderoReady(
    anatomy: AnatomyChoice | undefined,
    heart: HeartModel,
    tables: BeatTables,
  ): Promise<void> {
    this.roderoHeart(anatomy, heart, tables);
    await this.roderoLoad;
  }

  /** How the frames form the image, for the Dev panel. */
  imageFormationStatus(imageFormation: ImageFormationChoice | undefined): string {
    if (imageFormation !== 'cole') return 'red';
    if (this.coleFailure) return `convolución: error (${this.coleFailure})`;
    return this.coleLib ? 'convolución' : 'convolución (cargando…)';
  }

  /** What heart the frames draw, for the Dev panel. */
  anatomyStatus(anatomy: AnatomyChoice | undefined, isCpuBackend: boolean): string {
    if (anatomy !== 'rodero') return 'analítica';
    if (this.roderoFailure) return `Rodero: error (${this.roderoFailure})`;
    if (!this.roderoLib || !this.roderoVolume) return 'Rodero (cargando…)';
    return isCpuBackend ? 'Rodero (CPU)' : 'Rodero (GPU)';
  }
}
