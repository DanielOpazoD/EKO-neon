import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  focusOwnsKey,
  freezeKeyApplies,
  shortcutApplies,
  spaceToggles,
  typesText,
} from './shortcuts';

/** A focus target as the handlers read it: tag, input type, role and editability. */
function el(tagName: string, attrs: { type?: string; role?: string } = {}): EventTarget {
  return {
    tagName,
    type: attrs.type,
    isContentEditable: false,
    getAttribute: (k: string) => (k === 'role' ? (attrs.role ?? null) : null),
  } as unknown as EventTarget;
}

const key = (k: string, mods: { ctrl?: boolean; meta?: boolean; alt?: boolean } = {}) => ({
  key: k,
  ctrlKey: mods.ctrl ?? false,
  metaKey: mods.meta ?? false,
  altKey: mods.alt ?? false,
});

describe('keyboard shortcuts', () => {
  it('leaves Space to text fields and to the checkboxes and radios it alone toggles', () => {
    assert.equal(typesText(el('INPUT', { type: 'text' })), true);
    assert.equal(typesText(el('INPUT', { type: 'checkbox' })), false);
    assert.equal(spaceToggles(el('INPUT', { type: 'checkbox' })), true);
    assert.equal(spaceToggles(el('INPUT', { type: 'radio' })), true);
    assert.equal(spaceToggles(el('DIV', { role: 'checkbox' })), true);
    // a button is pressed by Enter: Space stays the freeze on it (decision 244)
    assert.equal(spaceToggles(el('BUTTON', { role: 'switch' })), false);
    assert.equal(spaceToggles(el('BUTTON')), false);
    assert.equal(spaceToggles(el('INPUT', { type: 'range' })), false);
  });

  it('freezes with Space only on the scanner screen, away from text and checkboxes', () => {
    assert.equal(freezeKeyApplies(el('BUTTON'), 'simulator'), true);
    assert.equal(freezeKeyApplies(el('INPUT', { type: 'range' }), 'simulator'), true);
    assert.equal(freezeKeyApplies(null, 'simulator'), true);
    assert.equal(freezeKeyApplies(el('INPUT', { type: 'checkbox' }), 'simulator'), false);
    assert.equal(freezeKeyApplies(el('TEXTAREA'), 'simulator'), false);
    // the impression sheet of the report: its checkboxes, and the page that Space scrolls
    assert.equal(freezeKeyApplies(el('INPUT', { type: 'checkbox' }), 'report'), false);
    assert.equal(freezeKeyApplies(el('BODY'), 'report'), false);
    assert.equal(freezeKeyApplies(el('BUTTON'), 'references'), false);
  });

  it('leaves Ctrl and ⌘ combinations to the browser, except undo in review mode', () => {
    assert.equal(shortcutApplies(key('c', { meta: true }), 'simulator'), false, '⌘C copies');
    assert.equal(shortcutApplies(key('c', { ctrl: true }), 'simulator'), false, 'Ctrl+C copies');
    assert.equal(shortcutApplies(key('=', { meta: true }), 'simulator'), false, '⌘+ zooms');
    assert.equal(shortcutApplies(key('-', { ctrl: true }), 'simulator'), false, 'Ctrl− zooms out');
    assert.equal(shortcutApplies(key('p', { ctrl: true }), 'simulator'), false, 'Ctrl+P prints');
    assert.equal(shortcutApplies(key('z', { meta: true }), 'simulator'), true);
    assert.equal(shortcutApplies(key('Z', { ctrl: true }), 'simulator'), true);
    assert.equal(shortcutApplies(key('c'), 'simulator'), true);
    assert.equal(shortcutApplies(key('ArrowLeft', { alt: true }), 'simulator'), true, 'Alt+← rocks');
    assert.equal(shortcutApplies(key('e', { alt: true }), 'simulator'), false, 'Alt+E is the browser menu');
  });

  it('takes the characters a keyboard writes with AltGr or Option as those characters', () => {
    // Spanish keyboards: «[» is AltGr+` on Windows (Ctrl+Alt on the page) and Option+` on macOS
    assert.equal(shortcutApplies(key('[', { ctrl: true, alt: true }), 'simulator'), true);
    assert.equal(shortcutApplies(key(']', { ctrl: true, alt: true }), 'simulator'), true);
    assert.equal(shortcutApplies(key('[', { alt: true }), 'simulator'), true);
    // a letter with AltGr is still not the letter's shortcut, nor is Ctrl alone with a character
    assert.equal(shortcutApplies(key('e', { ctrl: true, alt: true }), 'simulator'), false);
    assert.equal(shortcutApplies(key('[', { ctrl: true }), 'simulator'), false);
    assert.equal(shortcutApplies(key('+', { ctrl: true }), 'simulator'), false, 'Ctrl++ zooms in');
  });

  it('acts off the scanner screen only for the shortcuts sheet and Esc', () => {
    for (const screen of ['report', 'curriculum', 'progress', 'references']) {
      assert.equal(shortcutApplies(key('ArrowDown'), screen), false, `${screen}: the page scrolls`);
      assert.equal(shortcutApplies(key('p'), screen), false, `${screen}: no modality change unseen`);
      assert.equal(shortcutApplies(key('?'), screen), true);
      assert.equal(shortcutApplies(key('Escape'), screen), true);
    }
  });

  it('still yields the keys a focused widget handles itself', () => {
    assert.equal(focusOwnsKey(el('DIV', { role: 'tab' }), 'ArrowRight'), true);
    assert.equal(focusOwnsKey(el('BUTTON'), 'Enter'), true);
    assert.equal(focusOwnsKey(el('BUTTON'), 'c'), false);
  });
});
