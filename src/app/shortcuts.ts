import { useEffect } from 'react';
import { useSimStore } from './store';
import { modePolicy } from './modePolicy';

/**
 * Keyboard accelerators (spec 4.2). Every shortcut has a visible UI equivalent.
 * Q/E rotate · W/S pressure · arrows slide · Shift+arrows rock/tilt · Space freeze · C colour · M M-mode · P PW · X CW · 2 2D · T TDI · H hide/show torso
 */
export const SHORTCUTS: { keys: string; action: string }[] = [
  { keys: 'Q / E', action: 'Rotar sonda −/+ 3° (Shift: 15°)' },
  { keys: '← → ↑ ↓', action: 'Deslizar sonda 2 mm (Shift: 1 cm)' },
  { keys: 'Alt + ← →', action: 'Rock −/+ 3°' },
  { keys: 'Alt + ↑ ↓', action: 'Tilt (abanico) +/− 3°' },
  { keys: 'W / S', action: 'Presión +/−' },
  { keys: 'Espacio', action: 'Congelar / reanudar' },
  { keys: '2 · C · M · P · X · T', action: '2D · Color · modo M · PW · CW · TDI' },
  { keys: '[ / ]', action: 'Profundidad −/+ 1 cm' },
  { keys: '- / +', action: 'Ganancia −/+ 2 dB' },
  { keys: 'H', action: 'Mostrar/ocultar torso 3D' },
  { keys: 'R', action: 'Modo revisión: marcar la imagen para un informe' },
  {
    keys: 'Supr / Retroceso',
    action: 'Revisión: borrar el marcador seleccionado (Ctrl+Z deshace)',
  },
  { keys: '. / ,', action: 'Cine: cuadro siguiente/anterior (congelada)' },
  { keys: 'Esc', action: 'Cancelar la medición en curso' },
  { keys: '?', action: 'Mostrar u ocultar esta lista de atajos' },
];

/** The pointer on the 3D navigator (decision 188): listed in the shortcuts sheet, one short line on the torso itself. */
export const TORSO_GESTURES: { keys: string; action: string }[] = [
  { keys: 'Arrastrar la piel', action: 'Deslizar la sonda' },
  { keys: 'Arrastrar el marcador · rueda', action: 'Rotar la sonda (Shift + rueda: 10°)' },
  { keys: 'Shift + arrastrar', action: 'Rock' },
  { keys: 'Alt + arrastrar', action: 'Tilt (abanico)' },
  { keys: 'Botón derecho + arrastrar', action: 'Orbitar la cámara' },
  { keys: 'Ctrl/⌘ + rueda', action: 'Acercar o alejar la cámara' },
];

/** Roles whose widgets move with the arrow keys (WAI-ARIA composite widgets and the slider). */
const ARROW_ROLES = new Set([
  'tab',
  'tablist',
  'slider',
  'radio',
  'radiogroup',
  'listbox',
  'option',
  'menu',
  'menuitem',
  'menuitemradio',
  'menuitemcheckbox',
  'spinbutton',
  'combobox',
  'tree',
  'treeitem',
  'grid',
  'gridcell',
]);
const ARROW_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']);

/**
 * Whether the focused element handles this key itself, so the global shortcut must yield (decision 176): every key in a
 * form field or an editable element; the arrows, Home and End in a tab, a slider or another composite widget; Enter on
 * a button, a link or a summary. With focus on a tab ArrowRight moved the probe. Space is not decided here: it is the
 * freeze on the scanner's screen, but where text is typed and on the checkboxes Space alone toggles (`freezeKeyApplies`).
 */
export function focusOwnsKey(target: EventTarget | null, key: string): boolean {
  const t = target as HTMLElement | null;
  if (!t || typeof t.tagName !== 'string') return false;
  if (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA') return true;
  if (t.isContentEditable) return true;
  const role = t.getAttribute('role');
  if (role && ARROW_ROLES.has(role) && ARROW_KEYS.has(key)) return true;
  const pressable =
    t.tagName === 'BUTTON' || t.tagName === 'A' || t.tagName === 'SUMMARY' || role === 'button';
  return pressable && key === 'Enter';
}

/** Input types that hold no typed text: Space on them is not a character (a slider, a checkbox, a button). */
const NON_TEXT_INPUTS = new Set([
  'range',
  'checkbox',
  'radio',
  'button',
  'submit',
  'reset',
  'color',
  'file',
  'image',
]);

/** Whether Space types into the focused element (a text field, a text area, an editable element) or opens a select. */
export function typesText(target: EventTarget | null): boolean {
  const t = target as HTMLElement | null;
  if (!t || typeof t.tagName !== 'string') return false;
  if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable) return true;
  return t.tagName === 'INPUT' && !NON_TEXT_INPUTS.has((t as HTMLInputElement).type);
}

/**
 * Whether Space is the only key that toggles the focused control: a native checkbox or radio button, or an element with
 * that role that is not a button (Enter presses a button, never a checkbox). Taken as the freeze, the impression sheet
 * of the report could not be filled from the keyboard, and it weighs 20 % of the exam.
 */
export function spaceToggles(target: EventTarget | null): boolean {
  const t = target as HTMLElement | null;
  if (!t || typeof t.tagName !== 'string') return false;
  if (t.tagName === 'INPUT') {
    const type = (t as HTMLInputElement).type;
    return type === 'checkbox' || type === 'radio';
  }
  const role = t.getAttribute('role');
  return t.tagName !== 'BUTTON' && (role === 'checkbox' || role === 'radio' || role === 'switch');
}

/** Whether Space, pressed on `target`, is the freeze: on the scanner's screen, away from text and from the checkboxes. */
export function freezeKeyApplies(target: EventTarget | null, screen: string): boolean {
  return screen === 'simulator' && !typesText(target) && !spaceToggles(target);
}

/**
 * Whether a key press is one of the simulator's shortcuts. With ⌘, or with Ctrl alone, it is the browser's or the
 * system's (copy, zoom, print), except Ctrl/⌘ + Z in review mode: ⌘C switched to colour and resumed a frozen image, ⌘+
 * and ⌘− moved the gain. Alt with a letter or a digit is a menu accelerator (Alt+E opens the browser's menu on Windows).
 * But a character the keyboard writes with Option or AltGr — AltGr reaches the page as Ctrl+Alt — is that character:
 * on Spanish, German or French keyboards that is how «[» and «]» are typed. Alt also modifies the arrows (rock, tilt).
 * Off the scanner's screen only «?» and Esc act: the arrows moved a probe no one saw and kept the page from scrolling.
 */
export function shortcutApplies(
  e: Pick<KeyboardEvent, 'key' | 'ctrlKey' | 'metaKey' | 'altKey'>,
  screen: string,
): boolean {
  if (e.metaKey || (e.ctrlKey && !e.altKey))
    return screen === 'simulator' && (e.key === 'z' || e.key === 'Z');
  if (e.altKey && /^[a-z0-9]$/i.test(e.key)) return false;
  return screen === 'simulator' || e.key === '?' || e.key === 'Escape';
}

/**
 * Space freezes and resumes the scanner wherever the focus is on its screen (decision 244): a button, a tab, a slider,
 * the 3D torso, the console. One listener on the window, in the capture phase, so it runs before any other and is the
 * only one: it takes the default (the focused button is not pressed by the key) and stops the event there. Space stays
 * itself where text is typed, on a checkbox or a radio button, whose only key it is, and on the other screens, where
 * no image is in sight and Space scrolls the page. Holding the key toggles once.
 */
function useFreezeKey(): void {
  useEffect(() => {
    let taken = false;
    const isSpace = (e: KeyboardEvent) => e.key === ' ' || e.code === 'Space';
    const onDown = (e: KeyboardEvent) => {
      taken = false;
      if (!isSpace(e) || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
      if (!freezeKeyApplies(e.target, useSimStore.getState().ui.screen)) return;
      taken = true;
      e.preventDefault();
      e.stopPropagation();
      if (!e.repeat) useSimStore.getState().toggleFreeze();
    };
    // a button can also act on the release of Space: the release of a taken press is taken too
    const onUp = (e: KeyboardEvent) => {
      if (!taken || !isSpace(e)) return;
      taken = false;
      e.preventDefault();
    };
    window.addEventListener('keydown', onDown, true);
    window.addEventListener('keyup', onUp, true);
    return () => {
      window.removeEventListener('keydown', onDown, true);
      window.removeEventListener('keyup', onUp, true);
    };
  }, []);
}

export function useShortcuts(): void {
  useFreezeKey();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.isComposing || focusOwnsKey(e.target, e.key)) return;
      const s = useSimStore.getState();
      if (!shortcutApplies(e, s.ui.screen)) return;
      const big = e.shiftKey;
      const step = big ? 1 : 0.2;
      const ang = big ? 15 : 3;
      switch (e.key) {
        case 'q':
        case 'Q':
          s.nudgeProbe({ rotationDeg: -ang });
          break;
        case 'e':
        case 'E':
          s.nudgeProbe({ rotationDeg: ang });
          break;
        case 'w':
        case 'W':
          s.nudgeProbe({ pressure: 0.1 });
          break;
        case 's':
        case 'S':
          s.nudgeProbe({ pressure: -0.1 });
          break;
        case 'ArrowLeft':
          e.preventDefault();
          if (e.altKey) s.nudgeProbe({ rockDeg: -3 });
          else s.nudgeProbe({ u: -step });
          break;
        case 'ArrowRight':
          e.preventDefault();
          if (e.altKey) s.nudgeProbe({ rockDeg: 3 });
          else s.nudgeProbe({ u: step });
          break;
        case 'ArrowUp':
          e.preventDefault();
          if (e.altKey) s.nudgeProbe({ tiltDeg: 3 });
          else s.nudgeProbe({ v: step });
          break;
        case 'ArrowDown':
          e.preventDefault();
          if (e.altKey) s.nudgeProbe({ tiltDeg: -3 });
          else s.nudgeProbe({ v: -step });
          break;
        case 'r':
        case 'R': {
          if (modePolicy(s.mode).devToolsAllowed && s.ui.devMode) {
            const on = !s.ui.reviewMode;
            s.setUi({ reviewMode: on, consoleTab: on ? 'revisar' : 'adquirir' });
          }
          break;
        }
        case '2':
          s.setModality('2d');
          break;
        case 'c':
        case 'C':
          s.setModality(s.modality === 'color' ? '2d' : 'color');
          break;
        case 'm':
        case 'M':
          s.setModality(e.shiftKey ? 'cmm' : 'm-mode');
          break;
        case 'p':
        case 'P':
          s.setModality('pw');
          break;
        case 'x':
        case 'X':
          s.setModality('cw');
          break;
        case 't':
        case 'T':
          s.setModality('tdi');
          break;
        case '[':
          s.setSettings({ depthCm: s.settings.depthCm - 1 });
          break;
        case ']':
          s.setSettings({ depthCm: s.settings.depthCm + 1 });
          break;
        case '-':
          s.setSettings({ gainDb: s.settings.gainDb - 2 });
          break;
        case '+':
        case '=':
          s.setSettings({ gainDb: s.settings.gainDb + 2 });
          break;
        case 'h':
        case 'H':
          if (!s.ui.minimal) s.setUi({ showTorso: !s.ui.showTorso });
          break;
        case '.':
          if (s.frozen) s.setCineOffset(s.cineOffset + 1);
          break;
        case ',':
          if (s.frozen) s.setCineOffset(s.cineOffset - 1);
          break;
        case '?':
          s.setUi({ shortcutsOpen: !s.ui.shortcutsOpen });
          break;
        case 'Escape':
          if (s.ui.shortcutsOpen) s.setUi({ shortcutsOpen: false });
          else if (s.activeMeasurementId) s.setActiveMeasurement(null);
          else if (s.activeTool !== 'none') s.setActiveTool('none');
          else if (s.reviewLinkParentId) s.armReviewLink(null);
          else if (s.reviewSelectedId) s.selectReviewMarker(null);
          break;
        case 'Delete':
        case 'Backspace':
          if (s.ui.reviewMode && s.reviewSelectedId) s.removeReviewMarker(s.reviewSelectedId);
          else return;
          break;
        case 'z':
        case 'Z':
          if ((e.ctrlKey || e.metaKey) && s.ui.reviewMode && s.reviewUndo.length)
            s.undoReviewRemove();
          else return;
          break;
        default:
          return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
