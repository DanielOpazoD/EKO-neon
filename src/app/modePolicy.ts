export type ProductMode = 'sandbox' | 'guided' | 'exam';

export interface ModePolicy {
  /** The recognised view and its score: image HUD, measurement list and technique grade (hidden in exam). */
  showViewFeedback: boolean;
  /** Preset probe poses ("PLAX", "A4C"… buttons). */
  presetsEnabled: boolean;
  /** Hints toggle and hint content. */
  hintsEnabled: boolean;
  /** Curriculum / progress / references screens reachable. */
  learningScreensEnabled: boolean;
  /** Automatic curriculum task evaluation runs on each frame. */
  evaluateCurriculum: boolean;
  /** Physics overlay and dev panel may be shown. */
  devToolsAllowed: boolean;
  /**
   * A free caliper (no protocol id) may stand for a required measurement, matched to the closest true value. Not in exam:
   * that match lets a volley of random measurements score full marks (decision 235).
   */
  freeMeasurementsScored: boolean;
  /**
   * The 3D navigator shows the heart, its cut, the canonical windows and the axes. Not in exam: the torso stays, with
   * its skin and ribs, so the learner still sees where the probe is, but the heart cut by the plane would name the view.
   */
  navigatorAnatomy: boolean;
  /**
   * The impression is part of the total even when left blank, which then scores 0. Outside the exam a blank impression
   * leaves the total to acquisition and measurements, since practice grades it only when the learner asks; in the exam
   * a blank one used to lift the total, so an unsure learner did better by not answering.
   */
  impressionRequired: boolean;
  /** The exam draws its case at random when it starts: one the learner chose would make hiding its name pointless. */
  assignsCase: boolean;
}

/**
 * Whether the screen may name the case (decision 234). A case's title is its diagnosis («Estenosis aórtica severa…»), and
 * so are the measurements it requires: during an exam the image, the case selector, the report and the measurement list
 * said what the learner was asked to find. They name it again once the exam is finished, for the debrief.
 */
export function showsCaseIdentity(mode: ProductMode, examFinished: boolean): boolean {
  return mode !== 'exam' || examFinished;
}

/** Name shown for the case while its identity is hidden. */
export const EXAM_CASE_NAME = 'Caso de examen';

/**
 * A finished exam is closed, as its confirmation says: no acquiring, no measuring, no changing the impression. Only the
 * impression used to close: Space or a modality key resumed the live image, whose views then filled the report's table
 * of views while the recorded score left them out (decision 236), and a measurement taken after the finish joined the
 * report's list beside a score row that read «No medida». The image stays frozen as the finish left it, with its cine
 * to review, and the measurement list stays the one scored; changing the mode reopens the case for practice.
 */
export function examClosed(s: { mode: ProductMode; examFinished: boolean }): boolean {
  return s.mode === 'exam' && s.examFinished;
}

/** What each product mode hides or disables. Extracted so the rules live in one place. */
export function modePolicy(mode: ProductMode): ModePolicy {
  const exam = mode === 'exam';
  return {
    // the recognised view and its score would tell the learner what the exam asks them to recognise (decision 154)
    showViewFeedback: !exam,
    presetsEnabled: !exam,
    hintsEnabled: !exam,
    learningScreensEnabled: !exam,
    evaluateCurriculum: !exam,
    devToolsAllowed: !exam,
    freeMeasurementsScored: !exam,
    navigatorAnatomy: !exam,
    impressionRequired: exam,
    assignsCase: exam,
  };
}

/** The screens that teach (decision 244): closed while the exam runs (`learningScreensEnabled`). */
const LEARNING_SCREENS: readonly string[] = ['references', 'curriculum', 'progress'];

/**
 * The screen a mode shows for the one asked: a learning screen the mode closes gives way to the simulator. The exam
 * disabled their buttons, but an exam started from the progress screen kept it open, and its export listed the event
 * that names the case drawn.
 */
export function screenInMode<S extends string>(screen: S, mode: ProductMode): S | 'simulator' {
  return !modePolicy(mode).learningScreensEnabled && LEARNING_SCREENS.includes(screen)
    ? 'simulator'
    : screen;
}

/** The case's name in an exported file: its id, or a neutral tag while the exam hides it (decision 234). */
export function exportCaseTag(mode: ProductMode, examFinished: boolean, caseId: string): string {
  return showsCaseIdentity(mode, examFinished) ? caseId : 'examen';
}
