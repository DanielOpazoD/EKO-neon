import { useSimStore } from '@/app/store';

/** Section tooltips are teaching content — suppressed in exam mode like the other hints. */
export function useTip(): (t: string) => string | undefined {
  const training = useSimStore((s) => s.mode !== 'exam');
  return (t: string) => (training ? t : undefined);
}
