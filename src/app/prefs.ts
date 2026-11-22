/**
 * The interface preferences kept across sessions in localStorage, with the type each must have.
 * One schema serves the save and the load: a key saved is a key read back, and anything else in
 * storage — an old or hand-edited entry, a value of the wrong type — is dropped key by key
 * instead of being spread into the store.
 */
const SEGMENT_MODELS = new Set(['LV_AHA17', 'LV_16'] as const);
const CONSOLE_TABS = new Set(['adquirir', 'imagen', 'doppler', 'medir', 'lab', 'revisar'] as const);

const isBool = (v: unknown): v is boolean => typeof v === 'boolean';
const isSegmentModel = (v: unknown): v is 'LV_AHA17' | 'LV_16' =>
  typeof v === 'string' && SEGMENT_MODELS.has(v as 'LV_AHA17' | 'LV_16');
const isConsoleTab = (
  v: unknown,
): v is 'adquirir' | 'imagen' | 'doppler' | 'medir' | 'lab' | 'revisar' =>
  typeof v === 'string' &&
  CONSOLE_TABS.has(v as 'adquirir' | 'imagen' | 'doppler' | 'medir' | 'lab' | 'revisar');

const PERSISTED = {
  showTorso: isBool,
  showSkeleton: isBool,
  showHints: isBool,
  showEcg: isBool,
  devMode: isBool,
  tutorialDone: isBool,
  showHud: isBool,
  navSkin: isBool,
  navHeart: isBool,
  navChambers: isBool,
  navValves: isBool,
  navVessels: isBool,
  navAxes: isBool,
  navCut: isBool,
  navWindows: isBool,
  navSplit: isBool,
  navLabels: isBool,
  navSegments: isBool,
  imageSegments: isBool,
  segmentsOpen: isBool,
  segmentModel: isSegmentModel,
  reviewFreezeOnMark: isBool,
  railMini: isBool,
  minimal: isBool,
  roomMode: isBool,
  consoleTab: isConsoleTab,
} as const;

type GuardType<T> = T extends (v: unknown) => v is infer U ? U : never;

export type PersistedPrefKey = keyof typeof PERSISTED;
export const PERSISTED_PREF_KEYS = Object.keys(PERSISTED) as PersistedPrefKey[];
export type PersistedPrefs = { [K in PersistedPrefKey]: GuardType<(typeof PERSISTED)[K]> };

/** The valid persisted keys of an unknown value parsed from storage; invalid or unknown keys are left out. */
export function parsePrefs(raw: unknown): Partial<PersistedPrefs> {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return {};
  const src = raw as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of PERSISTED_PREF_KEYS) {
    const value = src[key];
    if (PERSISTED[key](value)) out[key] = value;
  }
  return out as Partial<PersistedPrefs>;
}

/** The persisted subset of the full preferences, ready to serialise. */
export function pickPrefs(ui: PersistedPrefs): PersistedPrefs {
  const out: Record<string, unknown> = {};
  for (const key of PERSISTED_PREF_KEYS) out[key] = ui[key];
  return out as PersistedPrefs;
}
