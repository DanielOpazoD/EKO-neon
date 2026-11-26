/**
 * GLSL mirrors of TypeScript scalar functions, translated back to JavaScript so that Node evaluates both copies.
 *
 * The scalar helpers the WebGL2 shaders share with the CPU tracer (those of glslGenerated.ts and a few in other shader
 * modules) are written twice, by hand: nothing generates the GLSL from the TypeScript. Each GLSL copy carries, on the
 * line before its definition, the annotation `// src/<path>.ts: <exportName>` naming its original. `glslMirrors` finds
 * the annotations in the shader sources and translates each annotated function, with the shader functions it calls and
 * its program's numeric #defines and consts, to JavaScript; cpuGpuParity.test.ts evaluates it against its original on
 * `mirrorSamples`. Only this test imports the module.
 *
 * The scalar subset of GLSL ES 3.00 is JavaScript once the declarations `float x =` become `let x =`, the constructors
 * `float(j)` become calls and the built-ins of §8 are defined on numbers. What falls outside it is reported with its
 * reason instead of translated: vector and matrix values, swizzles, arrays, structs, `out`/`inout` parameters, and the
 * names that are not numbers in the shader (the per-frame parameter texture P(i) and its defines, uniforms). Integer
 * division is not emulated (it would divide as floats): no mirror divides two ints.
 */

export type GlslScalar = number | boolean;

/** A GLSL function annotated as the mirror of a TypeScript export, with its JavaScript translation or why it has none. */
export interface GlslMirror {
  /** The TypeScript module (repository-relative) and export the annotation names. */
  tsPath: string;
  tsName: string;
  /** The GLSL definition that follows the annotation, and the types of its parameters. */
  glslName: string;
  paramTypes: string[];
  glsl?: (...args: GlslScalar[]) => GlslScalar;
  reason?: string;
  /** Numeric literals and constants of the function and of those it calls: the values its branches turn on. */
  landmarks: number[];
}

/** `// src/<path>.ts: <exportName>` on the line before a definition; groups: path, export, GLSL name. */
const MIRROR_ANNOTATION = /^[ \t]*\/\/[ \t]*(src\/\S+\.ts):[ \t]*(\w+)[ \t]*\r?\n[ \t]*\w+[ \t]+(\w+)\s*\(/gm;

/** GLSL ES 3.00 built-ins (§8) on the scalar genType, and the scalar constructors as `__float`, `__int`… */
const BUILTINS: Record<string, (...a: number[]) => GlslScalar> = {
  abs: Math.abs,
  sign: Math.sign,
  floor: Math.floor,
  ceil: Math.ceil,
  trunc: Math.trunc,
  round: Math.round,
  fract: (x) => x - Math.floor(x),
  mod: (x, y) => x - y * Math.floor(x / y),
  min: (a, b) => Math.min(a, b),
  max: (a, b) => Math.max(a, b),
  clamp: (x, lo, hi) => Math.min(Math.max(x, lo), hi),
  mix: (a, b, t) => a * (1 - t) + b * t,
  step: (edge, x) => (x < edge ? 0 : 1),
  smoothstep: (e0, e1, x) => {
    const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
    return t * t * (3 - 2 * t);
  },
  sqrt: Math.sqrt,
  inversesqrt: (x) => 1 / Math.sqrt(x),
  exp: Math.exp,
  exp2: (x) => 2 ** x,
  log: Math.log,
  log2: Math.log2,
  pow: Math.pow,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: (y, x) => (x === undefined ? Math.atan(y) : Math.atan2(y, x)),
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  radians: (deg) => (deg * Math.PI) / 180,
  degrees: (rad) => (rad * 180) / Math.PI,
  length: Math.abs,
  distance: (a, b) => Math.abs(a - b),
  dot: (a, b) => a * b,
  normalize: (x) => x / Math.abs(x),
  isnan: Number.isNaN,
  isinf: (x) => x === Infinity || x === -Infinity,
  __float: (x) => +x,
  __int: (x) => Math.trunc(+x),
  __uint: (x) => Math.trunc(+x) >>> 0,
  __bool: (x) => !!x,
};

const NUMBER = /^\(?\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[fFuU]?\s*\)?$/;
const LITERAL = /(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)/g;
const stripComments = (s: string): string => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
const numberOf = (literal: string): number => Number(literal.replace(/[()\sfFuU]/g, ''));

interface GlslDef {
  ret: string;
  name: string;
  params: string[];
  body: string;
  text: string;
}

/** Function definitions and top-level constants of one program; a name defined two different ways maps to null. */
function parseProgram(sources: readonly string[]) {
  const defs = new Map<string, GlslDef | null>();
  const consts = new Map<string, string | null>();
  const put = <T>(map: Map<string, T | null>, key: string, value: T, same: (a: T, b: T) => boolean) => {
    const old = map.get(key);
    map.set(key, old === undefined || (old !== null && same(old, value)) ? value : null);
  };
  for (const raw of sources) {
    const src = stripComments(raw);
    const structs = [...src.matchAll(/\bstruct\s+(\w+)/g)].map((m) => m[1]!);
    const types = ['void', 'float', 'int', 'uint', 'bool', '[iub]?vec[234]', 'mat[234]', ...structs].join('|');
    const header = new RegExp(`\\b(${types})\\s+(\\w+)\\s*\\(([^()]*)\\)\\s*\\{`, 'g');
    let outside = '';
    let last = 0;
    for (let m = header.exec(src); m; m = header.exec(src)) {
      let i = header.lastIndex;
      for (let depth = 1; depth > 0 && i < src.length; i++) depth += src[i] === '{' ? 1 : src[i] === '}' ? -1 : 0;
      const params = m[3]!.split(',').map((p) => p.trim()).filter((p) => p && p !== 'void');
      const body = src.slice(header.lastIndex, i - 1);
      put(defs, m[2]!, { ret: m[1]!, name: m[2]!, params, body, text: src.slice(m.index, i) }, (a, b) => a.text === b.text);
      outside += src.slice(last, m.index);
      last = header.lastIndex = i;
    }
    outside += src.slice(last);
    const same = (a: string, b: string) => a === b;
    for (const m of outside.matchAll(/^[ \t]*#define[ \t]+(\w+)[ \t]+(\S[^\n]*?)[ \t]*$/gm)) put(consts, m[1]!, m[2]!, same);
    for (const m of outside.matchAll(/\bconst\s+(?:(?:high|medium|low)p\s+)?\w+\s+(\w+)\s*=\s*([^;]+);/g))
      put(consts, m[1]!, m[2]!.trim(), same);
    for (const m of outside.matchAll(/\buniform\s+\w+\s+(\w+)/g)) put(consts, m[1]!, 'a uniform', same);
  }
  return { defs, consts };
}

/** Why a definition falls outside the scalar subset, or undefined when it is inside. */
function outsideSubset(d: GlslDef): string | undefined {
  if (!/^(?:float|int|uint|bool)$/.test(d.ret)) return `it returns ${d.ret}`;
  const scalarParam = /^(?:(?:in|const|highp|mediump|lowp)\s+)*(?:float|int|uint|bool)\s+\w+$/;
  for (const p of d.params) if (!scalarParam.test(p)) return `of its parameter "${p}"`;
  if (/\b(?:[iub]?vec[234]|mat[234](?:x[234])?|sampler\w*|struct)\b/.test(d.body)) return 'it uses vectors or matrices';
  if (/(?<!\d)\.\s*[A-Za-z_]/.test(d.body)) return 'it uses swizzles or struct members';
  if (d.body.includes('[')) return 'it uses arrays';
  if (/^\s*#/m.test(d.body)) return 'it holds preprocessor directives';
  if (/[&|^~]|<<|>>/.test(d.body.replace(/&&|\|\|/g, ''))) return 'it uses bitwise or ^^ operators';
  return undefined;
}

/** The definition as a JavaScript function declaration. */
function toJs(d: GlslDef): string {
  const params = d.params.map((p) => p.split(/\s+/).pop()!);
  const body = d.body
    .replace(/\b(?:const\s+)?(?:(?:high|medium|low)p\s+)?(?:float|int|uint|bool)\s+(?=[A-Za-z_]\w*\s*[=;,])/g, 'let ')
    .replace(/\b(float|int|uint|bool)\s*\(/g, '__$1(')
    .replace(/(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)[fFuU]\b/g, '$1');
  return `function ${d.name}(${params.join(', ')}) {${body}}`;
}

/** The annotated mirrors of one program (the sources compiled together), translated with that program's names. */
function programMirrors(program: readonly string[]): GlslMirror[] {
  const annotations = program.flatMap((s) => [...s.matchAll(MIRROR_ANNOTATION)]);
  if (!annotations.length) return [];
  const { defs, consts } = parseProgram(program);
  // the functions a body calls, other than the built-ins and the constructors
  const callees = (d: GlslDef): string[] =>
    [...d.body.matchAll(/(?<![.\w])([A-Za-z_]\w*)\s*\(/g)]
      .map((m) => m[1]!)
      .filter((n) => !(n in BUILTINS) && !/^(?:float|int|uint|bool|if|for|while|switch|return)$/.test(n));
  const closure = (name: string, seen = new Set<string>()): Set<string> => {
    if (seen.has(name)) return seen;
    seen.add(name);
    const d = defs.get(name);
    if (d) for (const c of callees(d)) closure(c, seen);
    return seen;
  };
  // why one definition, by itself, cannot be translated
  const ownReason = (name: string): string | undefined => {
    const d = defs.get(name);
    if (d === null) return `${name} is defined differently in two places`;
    if (!d) return `${name} is neither a scalar GLSL built-in nor a function of the program`;
    const subset = outsideSubset(d);
    if (subset) return `${name} is outside the scalar subset: ${subset}`;
    for (const [, w] of d.body.matchAll(/(?<![.\w])([A-Za-z_]\w*)\b/g)) {
      const v = consts.get(w!);
      if (v === null) return `${w} is defined differently in two places`;
      if (v !== undefined && !NUMBER.test(v)) return `${name} reads ${w}, ${v}`;
    }
    try {
      new Function(...Object.keys(BUILTINS), toJs(d));
    } catch (e) {
      return `JavaScript rejects the translation of ${name}: ${(e as Error).message}`;
    }
    return undefined;
  };

  const mirrors: GlslMirror[] = annotations.map(([, tsPath, tsName, glslName]) => {
    const deps = closure(glslName!);
    const landmarks = new Set<number>([0, 1]);
    for (const n of deps) {
      const body = defs.get(n)?.body ?? '';
      for (const [, lit] of body.matchAll(LITERAL)) landmarks.add(Number(lit));
      for (const [, w] of body.matchAll(/(?<![.\w])([A-Za-z_]\w*)\b/g)) {
        const v = consts.get(w!);
        if (v && NUMBER.test(v)) landmarks.add(numberOf(v));
      }
    }
    const reason = [...deps].map(ownReason).find(Boolean);
    return {
      tsPath: tsPath!,
      tsName: tsName!,
      glslName: glslName!,
      paramTypes: (defs.get(glslName!)?.params ?? []).map((p) => p.split(/\s+/).at(-2) ?? ''),
      reason,
      landmarks: [...landmarks],
    };
  });

  const translated = new Set(mirrors.filter((m) => !m.reason).flatMap((m) => [...closure(m.glslName)]));
  const constLines = [...consts]
    .filter((e): e is [string, string] => e[1] !== null && NUMBER.test(e[1]))
    .map(([k, v]) => `const ${k} = ${numberOf(v)};`);
  const module = [
    '"use strict";',
    ...constLines,
    ...[...translated].map((n) => toJs(defs.get(n)!)),
    `return { ${[...translated].join(', ')} };`,
  ].join('\n');
  const lib = new Function(...Object.keys(BUILTINS), module)(...Object.values(BUILTINS)) as Record<
    string,
    (...args: GlslScalar[]) => GlslScalar
  >;
  for (const m of mirrors) if (!m.reason) m.glsl = lib[m.glslName];
  return mirrors;
}

/**
 * Every annotated mirror in the shader sources. The sources without `#version` are the pieces the image-formation
 * programs are assembled from (each is GLSL_COMMON plus the heart, the thorax or a pass), so they are read as one
 * program, where a name two pieces define differently (each pass's main) is left untranslated; each `#version` source
 * (Doppler, image chain) is a program of its own, with its own constants.
 */
export function glslMirrors(sources: readonly string[]): GlslMirror[] {
  const isProgram = (s: string) => /^\s*#version/.test(s);
  const programs = [sources.filter((s) => !isProgram(s)), ...sources.filter(isProgram).map((s) => [s])];
  const byName = new Map<string, GlslMirror>();
  for (const m of programs.flatMap(programMirrors)) {
    const old = byName.get(m.glslName);
    if (!old || (old.reason && !m.reason)) byName.set(m.glslName, m);
  }
  return [...byName.values()];
}

/**
 * Largest difference between a TypeScript original and its GLSL mirror that is not a divergence, relative to
 * max(1, |original|): 2⁻²⁰ ≈ 9.5e-7. The translation runs in double precision, where two copies of one formula can
 * differ only by reassociation and by constants printed with fewer digits. The shader's PI, 3.14159265358979, is the
 * one such constant today (the same float32 as Math.PI): through it four mirrors differ by 1.4e-10 at most, and the
 * other fifty agree to the last bit. The GPU evaluates in float32, whose unit roundoff is 2⁻²⁴ ≈ 6e-8: sixteen of them
 * at the result's scale (1 cm or 1 unit at least) are below what the GPU can show, and a coefficient, branch or
 * constant that moves the result by more is a divergence. The GPU's own float32 rounding is not modelled here.
 */
export const MIRROR_TOLERANCE = 2 ** -20;

/** Difference between the original's and the mirror's value, relative to max(1, |original|); NaN equals NaN. */
export function mirrorDeviation(original: GlslScalar, mirror: GlslScalar): number {
  if (original === mirror || (Number.isNaN(original) && Number.isNaN(mirror))) return 0;
  if (typeof original !== 'number' || typeof mirror !== 'number') return Infinity;
  if (!Number.isFinite(original) || !Number.isFinite(mirror)) return Infinity;
  return Math.abs(original - mirror) / Math.max(1, Math.abs(original));
}

/**
 * Deterministic arguments for a mirror. Each argument mixes broad ranges, which hold the inputs these functions see in
 * use (fractions and cosines ±1.5, angles and heart coordinates ±8, thorax coordinates ±40 cm), with the function's
 * landmarks, their negatives, sums and differences, exact and jittered: the values where its branches and clamps turn.
 * An int parameter gets the value rounded, a bool one its sign.
 */
export function mirrorSamples(m: GlslMirror, count: number): GlslScalar[][] {
  let seed = [...m.glslName].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261);
  const random = (): number => {
    // mulberry32
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const marks = new Set<number>();
  for (const a of m.landmarks) {
    marks.add(a).add(-a);
    for (const b of m.landmarks) marks.add(a + b).add(a - b);
  }
  const landmarks = [...marks];
  const out: GlslScalar[][] = [];
  for (let s = 0; s < count; s++) {
    out.push(
      m.paramTypes.map((type) => {
        const r = random();
        const u = 2 * random() - 1;
        const l = landmarks[Math.floor(random() * landmarks.length)]!;
        const jittered = l + 0.1 * u * Math.max(0.25, Math.abs(l));
        const v = r < 0.3 ? 1.5 * u : r < 0.45 ? 8 * u : r < 0.6 ? 40 * u : r < 0.75 ? l : jittered;
        return type === 'bool' ? v > 0 : type === 'int' ? Math.round(v) : type === 'uint' ? Math.abs(Math.round(v)) : v;
      }),
    );
  }
  return out;
}
