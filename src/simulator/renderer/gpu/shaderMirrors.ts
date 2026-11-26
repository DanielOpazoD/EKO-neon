/**
 * GLSL mirrors of TypeScript functions, translated back to JavaScript so that Node evaluates both copies.
 *
 * The shaders' copies of the CPU tracer's functions are written by hand: nothing generates the GLSL from the TypeScript.
 * Each copy carries before its definition the annotation `// src/<path>.ts: <exportName>` naming its original (blank
 * lines, other comments and a precision qualifier may stand between them). Inside a function the annotation names the
 * TypeScript function that some of its statements mirror: the block `{ … }` it stands before, or the run of statements
 * from it to the line `// end of <exportName>`, whose declarations the function reads after it (the classifier's blocks
 * and the wiring between them: the point after the swing, the root's coordinates, the atria's outer faces for the sac).
 * `glslMirrors` translates each annotated function or block, with the shader functions it calls and its program's
 * #defines and consts, to JavaScript; cpuGpuParity.test.ts evaluates it against its original. Only the tests import
 * this module.
 *
 * The translation is typed as GLSL ES 3.00 is. A value is a float, int or bool, a vec2-4 or ivec2-4 (an array of numbers
 * here), a struct (an object) or a constant array of scalars (`const float PV_COURSE[12] = float[12](…)`, read by
 * index), and each operator and built-in (§5, §8) is emitted for the types of its operands: component-wise on vectors,
 * scalars broadcast; int arithmetic on 32 bits in two's complement (§4.1.3: +, − and × keep the low bits, / and %
 * truncate, the bitwise operators act on the bits); vectors and structs copied where GLSL copies values (declarations,
 * assignments, in parameters), out and inout parameters copied back on return; a swizzle, of a swizzle too, read from and
 * written to the vector beneath it. A variable declared without a value and an out parameter on entry are undefined: NaN
 * through a float, field by field in a struct. Where GLSL leaves the result undefined — an int division by zero or one
 * that overflows, a negative operand of %, a shift by a negative count or by 32 or more, int() of a float beyond 32
 * bits, an index beyond a vector or an array, pow, log, log2, sqrt and inversesqrt outside their domain (§8.2) — the
 * translation raises GlslUndefined instead of giving JavaScript's number.
 * A texture read (texelFetch of a sampler uniform) is bound: `bind` receives, under the sampler's name, the JavaScript
 * that reads the texel from the data the CPU uploads, and may replace any function or scalar built-in by name the same
 * way (P(i) by the packed array itself, a CPU approximation the comparison leaves out). What the translation would get
 * wrong is reported with its reason instead: uint (its arithmetic wraps modulo 2³²), round (GLSL leaves the rounding of
 * halves to the implementation), filtered texture reads, bitwise operators on floats, the other arrays, matrices,
 * boolean vectors, implicit conversions (GLSL ES 3.00 has none), a function that returns a struct, and the uniforms
 * other than samplers and the shader inputs a body reads.
 */

/** A value of a mirror's arguments or result: a float, int or bool, or a vector's components. */
export type GlslValue = number | boolean | number[];

/**
 * JavaScript that stands for what the GPU reads and for GLSL functions, by name. Under a sampler uniform's name, the
 * texel its texelFetch reads, from the data the CPU uploads: (coordinates, level) → the four components as the shader
 * sees them (required for each sampler a translation reads). Under a function's name, JavaScript that replaces it: the
 * CPU's own approximation of it, which the comparison leaves out, or a faster read of the same data (P(i) as the packed
 * array itself). A function with out or inout parameters returns its value and then each of those, in an array.
 */
export type GlslBindings = Record<string, (...args: GlslValue[]) => GlslValue | GlslValue[]>;

/**
 * What GLSL ES 3.00 leaves undefined, met by a translation as it runs (an int division by zero, a shift beyond the
 * int's bits, pow of a negative base, an out-of-range vector index…): the GPU may give anything there, so the sample
 * tells nothing of it, and agrees with nothing.
 */
export class GlslUndefined extends RangeError {}

/** One run of a block mirror. */
export interface GlslBlockRun {
  /** What the block returns from its function, or undefined when it runs to its end. */
  returned: unknown;
  /**
   * The block's inputs as it leaves them (the variables of its function it assigns, the structs it writes) and, for a run
   * of statements, the variables it declares (undefined where it returned before declaring them).
   */
  values: Record<string, unknown>;
}

/** A GLSL function or block annotated as the mirror of a TypeScript export, with its translation or why it has none. */
export interface GlslMirror {
  /** The TypeScript module (repository-relative) and export the annotation names. */
  tsPath: string;
  tsName: string;
  /**
   * A function, or statements of one, named `function{export}`: a block `{ … }`, or a run of statements that ends at the
   * line `// end of <export>` and whose declarations its function reads after it.
   */
  kind: 'function' | 'block';
  glslName: string;
  /** The types of the parameters; for a block, of its inputs. */
  paramTypes: string[];
  /** A block's inputs: the parameters and variables of its function that it reads, which its runner takes by name. */
  inputs: string[];
  /** The variables a run of statements declares for its function, which it leaves in `values`; none for a `{ … }`. */
  outputs: string[];
  /** The sampler uniforms whose texels the translation reads (uParams, uNoise): what `bind` must supply. */
  reads: string[];
  /** The translation, for a function that reads no texture. */
  glsl?: (...args: GlslValue[]) => GlslValue;
  /** The translation of a function, with the bindings of its samplers and of the functions the caller replaces. */
  bind?: (bindings: GlslBindings) => (...args: GlslValue[]) => GlslValue;
  /** The translation of a block, bound likewise, run with its inputs by name. */
  bindBlock?: (bindings: GlslBindings) => (inputs: Record<string, unknown>) => GlslBlockRun;
  reason?: string;
  /** Numeric literals and constants of the function and of those it calls: the values its branches turn on. */
  landmarks: number[];
  /** The literal indices i of the P(i) it reads, whose packed values are the landmarks of a frame. */
  parameterReads: number[];
}

/**
 * The annotation as the translation reads it, wherever it stands in its line; what follows it (past blank lines and
 * comments) is the definition or block it names. Groups: TypeScript path, export.
 */
export const MIRROR_ANNOTATION = /\/\/[ \t]*(src\/[^\s:]+\.ts)[ \t]*:[ \t]*(\w+)/g;

// ------------------------------------------------------------------------------------------------ errors and names

class Untranslatable extends Error {}
const fail = (reason: string): never => {
  throw new Untranslatable(reason);
};

/** GLSL identifiers that JavaScript reserves or that the generated code reads as globals. */
const JS_RESERVED = new Set(
  (
    'arguments await catch class debugger delete enum eval export extends finally function implements import ' +
    'instanceof interface let new null package private protected public static super this throw try typeof var ' +
    'with yield Math Object Array Number NaN Infinity undefined'
  ).split(' '),
);
const js = (name: string): string => (JS_RESERVED.has(name) ? `$${name}` : name);

// ------------------------------------------------------------------------------------------------ built-ins

const undefinedIn = (what: string): never => {
  throw new GlslUndefined(`${what}: undefined in GLSL ES 3.00`);
};

/**
 * GLSL ES 3.00 built-ins (§8) on scalars; their vector forms apply them component by component. The exponential
 * functions are undefined outside their domain (§8.2), where JavaScript gives a number (pow(−2, 2) = 4, log(0) = −∞)
 * that the GPU need not: there the translation raises GlslUndefined.
 */
const BUILTINS: Record<string, (...a: number[]) => number | boolean> = {
  abs: Math.abs,
  sign: Math.sign,
  floor: Math.floor,
  ceil: Math.ceil,
  trunc: Math.trunc,
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
  sqrt: (x) => (x < 0 ? undefinedIn(`sqrt(${x})`) : Math.sqrt(x)),
  inversesqrt: (x) => (x <= 0 ? undefinedIn(`inversesqrt(${x})`) : 1 / Math.sqrt(x)),
  exp: Math.exp,
  exp2: (x) => 2 ** x,
  log: (x) => (x <= 0 ? undefinedIn(`log(${x})`) : Math.log(x)),
  log2: (x) => (x <= 0 ? undefinedIn(`log2(${x})`) : Math.log2(x)),
  pow: (x, y) => (x < 0 || (x === 0 && y <= 0) ? undefinedIn(`pow(${x}, ${y})`) : Math.pow(x, y)),
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
  isnan: Number.isNaN,
  isinf: (x) => x === Infinity || x === -Infinity,
};
/** The built-ins whose int overloads (§8.3) are emitted as their float forms. */
const INT_BUILTINS = new Set(['abs', 'sign', 'min', 'max', 'clamp']);

type Vec = number[];
const componentWise =
  (f: (...a: number[]) => number | boolean) =>
  (...a: (number | Vec)[]): Vec => {
    const n = (a.find(Array.isArray) as Vec).length;
    return Array.from({ length: n }, (_, i) => f(...a.map((x) => (Array.isArray(x) ? x[i]! : x))) as number);
  };
const vop =
  (f: (x: number, y: number) => number) =>
  (a: number | Vec, b: number | Vec): Vec =>
    typeof a === 'number'
      ? (b as Vec).map((y) => f(a, y))
      : typeof b === 'number'
        ? a.map((x) => f(x, b))
        : a.map((x, i) => f(x, b[i]!));
const length = (v: Vec): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

/**
 * int arithmetic as GLSL ES 3.00 does it on a highp int, 32 bits in two's complement (§4.1.3): addition, subtraction
 * and multiplication keep the low 32 bits of the exact result, the bitwise operators act on those bits (§5.9); the rest
 * is undefined (a division by zero or one that overflows, a negative operand of %, a shift by a negative count or by 32
 * or more). Every int a translation holds is an int32 — int() of a float out of range is undefined too — so | 0 and
 * Math.imul are exact.
 */
const INT_OPS: Record<string, (a: number, b: number) => number> = {
  '+': (a, b) => (a + b) | 0,
  '-': (a, b) => (a - b) | 0,
  '*': Math.imul,
  '/': (a, b) => (b === 0 ? undefinedIn('int division by zero') : a === -2147483648 && b === -1 ? undefinedIn(`${a} / ${b}, which overflows`) : (a / b) | 0),
  '%': (a, b) => (b === 0 || a < 0 || b < 0 ? undefinedIn(`${a} % ${b}`) : a % b),
  '&': (a, b) => a & b,
  '|': (a, b) => a | b,
  '^': (a, b) => a ^ b,
  '<<': (a, b) => (b < 0 || b > 31 ? undefinedIn(`a shift by ${b}`) : a << b),
  '>>': (a, b) => (b < 0 || b > 31 ? undefinedIn(`a shift by ${b}`) : a >> b),
};

/** A copy of a value, as GLSL copies vectors and structs on assignment, initialisation and in parameters. */
const clone = (v: unknown): unknown =>
  Array.isArray(v)
    ? v.slice()
    : v !== null && typeof v === 'object'
      ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, clone(x)]))
      : v;

/** What the generated code calls besides the built-ins. */
const RUNTIME: Record<string, unknown> = {
  __idiv: INT_OPS['/'],
  __imod: INT_OPS['%'],
  __shl: INT_OPS['<<'],
  __shr: INT_OPS['>>'],
  __iv: {
    ...Object.fromEntries(Object.entries(INT_OPS).map(([op, f]) => [op, vop(f)])),
    neg: (v: Vec): Vec => v.map((x) => -x | 0),
    not: (v: Vec): Vec => v.map((x) => ~x),
  },
  __toInt: (x: number): number => (x > -2147483649 && x < 2147483648 ? x | 0 : undefinedIn(`int(${x})`)),
  __ix: (n: number, i: number): number => (i >= 0 && i < n ? i : undefinedIn(`index ${i} of a vector or an array of ${n}`)),
  __clone: clone,
  /**
   * A struct assigned whole: the target object takes the value's fields (an out parameter is the caller's object). The
   * value is copied before the target is emptied: assigned to itself (a = a, a = c ? a : b, one struct passed to two
   * inout parameters) it would otherwise be emptied with it.
   */
  __assignStruct: (target: unknown, value: unknown): unknown => {
    const copy = clone(value);
    if (target === null || typeof target !== 'object') return copy;
    for (const k of Object.keys(target)) delete (target as Record<string, unknown>)[k];
    return Object.assign(target, copy);
  },
  __add: vop((x, y) => x + y),
  __sub: vop((x, y) => x - y),
  __mul: vop((x, y) => x * y),
  __div: vop((x, y) => x / y),
  __neg: (v: Vec): Vec => v.map((x) => -x),
  __eq: (a: Vec, b: Vec): boolean => a.every((x, i) => x === b[i]),
  __copy: (v: Vec): Vec => v.slice(),
  __swz: (v: Vec, idx: number[]): Vec => idx.map((i) => v[i]!),
  __swzSet: (v: Vec, idx: number[], w: Vec): Vec => {
    idx.forEach((i, k) => (v[i] = w[k]!));
    return w;
  },
  __splat: (n: number, x: number): Vec => new Array<number>(n).fill(x),
  __cat: (n: number, ...parts: (number | Vec)[]): Vec => {
    const out = parts.flat();
    if (out.length < n) throw new RangeError(`vec${n} constructed from ${out.length} components`);
    return out.slice(0, n);
  },
  __undef: (n: number): Vec => new Array<number>(n).fill(NaN),
  // the geometric functions (§8.5) on a float and on a vector
  __length: (v: number | Vec): number => (typeof v === 'number' ? Math.abs(v) : length(v)),
  __dot: (a: number | Vec, b: number | Vec): number =>
    typeof a === 'number' ? a * (b as number) : a.reduce((s, x, i) => s + x * (b as Vec)[i]!, 0),
  __cross: (a: Vec, b: Vec): Vec => [
    a[1]! * b[2]! - a[2]! * b[1]!,
    a[2]! * b[0]! - a[0]! * b[2]!,
    a[0]! * b[1]! - a[1]! * b[0]!,
  ],
  __normalize: (v: number | Vec): number | Vec => {
    if (typeof v === 'number') return v / Math.abs(v);
    const l = length(v);
    return v.map((x) => x / l);
  },
  __v: Object.fromEntries(Object.entries(BUILTINS).map(([k, f]) => [k, componentWise(f)])),
};
const RUNTIME_NAMES = [...Object.keys(RUNTIME), ...Object.keys(BUILTINS)];
const RUNTIME_VALUES = [...Object.values(RUNTIME), ...Object.values(BUILTINS)];

// ------------------------------------------------------------------------------------------------ types

/** vec2-4 (float components) and ivec2-4 (int components): arrays of numbers here. */
const VEC = /^(i?)vec([234])$/;
const isVec = (t: string): boolean => VEC.test(t);
const sizeOf = (t: string): number => Number(VEC.exec(t)?.[2] ?? 1);
/** The type of a vector's components; a scalar type is its own. */
const componentOf = (t: string): string => (isVec(t) ? (t[0] === 'i' ? 'int' : 'float') : t);
const vecOf = (component: string, n: number): string => (n === 1 ? component : `${component === 'int' ? 'i' : ''}vec${n}`);
const isIntType = (t: string): boolean => componentOf(t) === 'int';
/** A constant array of scalars, `float[12]`: the only array the translation holds, read by index. */
const ARRAY = /^(float|int|bool)\[(\d+)\]$/;
const isArray = (t: string): boolean => ARRAY.test(t);
const BUILTIN_TYPES =
  /^(?:void|float|int|uint|bool|[iub]?vec[234]|mat[234](?:x[234])?|[iu]?sampler\w+)$/;
const SWIZZLE_SETS = ['xyzw', 'rgba', 'stpq'];
/** The types a translated value may have, besides the program's structs. */
function checkValueType(t: string, what: string, structs: Map<string, unknown>): void {
  if (t === 'uint' || /^uvec/.test(t)) fail(`${what} is uint, whose arithmetic wraps modulo 2³² and is not emulated`);
  if (/^bvec/.test(t)) fail(`${what} is a boolean vector (${t})`);
  if (/^mat/.test(t)) fail(`${what} is a matrix (${t})`);
  if (/sampler/.test(t)) fail(`${what} is a sampler`);
  if (!/^(?:float|int|bool|void)$/.test(t) && !isVec(t) && !structs.has(t)) fail(`${what} has the type ${t}`);
}
/**
 * JavaScript for a value GLSL leaves undefined (a variable declared without one, an out parameter on entry, §6.1.1): a
 * float reads as NaN through what it enters, a vector holds NaN, a struct such fields. An int so left reads as 0 once an
 * int operator touches it (| 0), where the GPU may read anything: that case is not flagged.
 */
function undefinedOf(type: string, structs: Map<string, Map<string, string> | null>): string {
  if (isVec(type)) return `__undef(${sizeOf(type)})`;
  const fields = structs.get(type);
  if (fields) return `{ ${[...fields].map(([f, t]) => `${f}: ${undefinedOf(t, structs)}`).join(', ')} }`;
  return 'undefined';
}

// ------------------------------------------------------------------------------------------------ lexer

interface Tok {
  k: 'num' | 'id' | 'op';
  v: string;
  /** Offset of the token in its source (of the macro's use, for the tokens a macro expands to). */
  at: number;
}
const LEXEME =
  /\s+|(0[xX][\da-fA-F]+[uU]?|\d+\.\d*(?:[eE][-+]?\d+)?[fF]?|\.\d+(?:[eE][-+]?\d+)?[fF]?|\d+[eE][-+]?\d+[fF]?|\d+[uU]?)|([A-Za-z_]\w*)|(<<=|>>=|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\^\^|[-+*/%&|^]=|[-+*/%<>=!~&|^?:;,.(){}[\]])/y;

/** The tokens of `text` (at offset `base` of its source), object-like macros expanded where they are used. */
function lex(text: string, base: number, macros: Map<string, Tok[] | null>): Tok[] {
  const out: Tok[] = [];
  const expand = (t: Tok, guard: Set<string>): void => {
    const body = t.k === 'id' && !guard.has(t.v) ? macros.get(t.v) : undefined;
    if (!body) {
      out.push(t);
      return;
    }
    guard.add(t.v);
    for (const u of body) expand({ ...u, at: t.at }, guard);
    guard.delete(t.v);
  };
  let i = 0;
  while (i < text.length) {
    LEXEME.lastIndex = i;
    const m = LEXEME.exec(text);
    if (!m) fail(`it holds "${text[i]}", which is no GLSL token`);
    i = LEXEME.lastIndex;
    const at = base + m!.index;
    if (m![1]) out.push({ k: 'num', v: m![1], at });
    else if (m![2]) expand({ k: 'id', v: m![2], at }, new Set());
    else if (m![3]) out.push({ k: 'op', v: m![3], at });
  }
  return out;
}

function literal(v: string): { value: number; type: string } {
  if (/[uU]$/.test(v)) return { value: Number(v.slice(0, -1)), type: 'uint' };
  // a hexadecimal int literal is a bit pattern (0xFFFFFFFF is −1)
  if (/^0[xX]/.test(v)) return { value: parseInt(v, 16) | 0, type: 'int' };
  if (/[.eEfF]/.test(v)) return { value: Number(v.replace(/[fF]$/, '')), type: 'float' };
  if (/^0\d/.test(v)) return { value: parseInt(v, 8), type: 'int' };
  return { value: Number(v), type: 'int' };
}

// ------------------------------------------------------------------------------------------------ parser

type Expr =
  | { k: 'lit'; value: number | boolean; type: string; at: number }
  | { k: 'id'; name: string; at: number }
  | { k: 'call'; name: string; args: Expr[]; at: number }
  | { k: 'dot'; e: Expr; name: string; at: number }
  | { k: 'index'; e: Expr; i: Expr; at: number }
  | { k: 'unary'; op: string; e: Expr; at: number }
  | { k: 'update'; op: string; prefix: boolean; e: Expr; at: number }
  | { k: 'binary'; op: string; a: Expr; b: Expr; at: number }
  | { k: 'cond'; c: Expr; a: Expr; b: Expr; at: number }
  | { k: 'assign'; op: string; target: Expr; e: Expr; at: number }
  | { k: 'seq'; a: Expr; b: Expr; at: number }
  | { k: 'array'; type: string; size?: number; args: Expr[]; at: number };
/** A statement, with the offsets in its source where it starts (`at`) and where its last token ends (`end`). */
type Stmt = (
  | { k: 'decl'; type: string; isConst: boolean; vars: { name: string; array: boolean; init?: Expr }[]; at: number }
  | { k: 'expr'; e: Expr; at: number }
  | { k: 'block'; body: Stmt[]; at: number }
  | { k: 'if'; c: Expr; t: Stmt; f?: Stmt; at: number }
  | { k: 'for'; init?: Stmt; c?: Expr; step?: Expr; body: Stmt; at: number }
  | { k: 'while'; c: Expr; body: Stmt; at: number }
  | { k: 'do'; body: Stmt; c: Expr; at: number }
  | { k: 'return'; e?: Expr; at: number }
  | { k: 'jump'; op: string; at: number }
) & { end?: number };

const BINARY_POWER: Record<string, number> = {
  '||': 4,
  '^^': 5,
  '&&': 6,
  '|': 7,
  '^': 8,
  '&': 9,
  '==': 10,
  '!=': 10,
  '<': 11,
  '>': 11,
  '<=': 11,
  '>=': 11,
  '<<': 12,
  '>>': 12,
  '+': 13,
  '-': 13,
  '*': 14,
  '/': 14,
  '%': 14,
};
const ASSIGN = new Set(['=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']);
const QUALIFIERS = new Set(['const', 'highp', 'mediump', 'lowp', 'precise', 'invariant', 'in', 'flat', 'smooth']);

/** Statements and expressions of a function body (GLSL ES 3.00 §5, §6). */
class Parser {
  private i = 0;
  constructor(
    private readonly toks: Tok[],
    private readonly isType: (name: string) => boolean,
  ) {}
  private peek(o = 0): Tok | undefined {
    return this.toks[this.i + o];
  }
  private at(): number {
    return this.peek()?.at ?? this.toks[this.toks.length - 1]?.at ?? 0;
  }
  private eat(v: string): boolean {
    if (this.peek()?.v !== v) return false;
    this.i++;
    return true;
  }
  private expect(v: string): void {
    if (!this.eat(v)) fail(`it does not parse: "${v}" expected before "${this.peek()?.v ?? 'the end'}"`);
  }
  private name(): string {
    const t = this.peek();
    if (t?.k !== 'id') fail(`it does not parse: a name expected before "${t?.v ?? 'the end'}"`);
    this.i++;
    return t!.v;
  }
  statements(): Stmt[] {
    const out: Stmt[] = [];
    while (this.peek()) out.push(this.statement());
    return out;
  }
  private startsDeclaration(): boolean {
    let o = 0;
    while (this.peek(o)?.k === 'id' && QUALIFIERS.has(this.peek(o)!.v)) o++;
    const t = this.peek(o),
      n = this.peek(o + 1);
    return t?.k === 'id' && this.isType(t.v) && (n?.k === 'id' || n?.v === '[');
  }
  private statement(): Stmt {
    const s = this.statementOnly();
    const last = this.toks[this.i - 1];
    s.end = last ? last.at + last.v.length : s.at;
    return s;
  }
  private statementOnly(): Stmt {
    const at = this.at();
    const t = this.peek()!;
    if (this.eat('{')) {
      const body: Stmt[] = [];
      while (!this.eat('}')) {
        if (!this.peek()) fail('it does not parse: "}" expected');
        body.push(this.statement());
      }
      return { k: 'block', body, at };
    }
    if (this.eat(';')) return { k: 'block', body: [], at };
    if (t.k === 'id') {
      switch (t.v) {
        case 'if': {
          this.i++;
          this.expect('(');
          const c = this.expression();
          this.expect(')');
          const then = this.statement();
          return { k: 'if', c, t: then, f: this.eat('else') ? this.statement() : undefined, at };
        }
        case 'for': {
          this.i++;
          this.expect('(');
          const init = this.eat(';') ? undefined : this.startsDeclaration() ? this.declaration() : this.exprStatement();
          const c = this.peek()?.v === ';' ? undefined : this.expression();
          this.expect(';');
          const step = this.peek()?.v === ')' ? undefined : this.expression();
          this.expect(')');
          return { k: 'for', init, c, step, body: this.statement(), at };
        }
        case 'while': {
          this.i++;
          this.expect('(');
          const c = this.expression();
          this.expect(')');
          return { k: 'while', c, body: this.statement(), at };
        }
        case 'do': {
          this.i++;
          const body = this.statement();
          if (!this.eat('while')) fail('it does not parse: "while" expected after do');
          this.expect('(');
          const c = this.expression();
          this.expect(')');
          this.expect(';');
          return { k: 'do', body, c, at };
        }
        case 'return': {
          this.i++;
          const e = this.peek()?.v === ';' ? undefined : this.expression();
          this.expect(';');
          return { k: 'return', e, at };
        }
        case 'break':
        case 'continue':
        case 'discard':
          this.i++;
          this.expect(';');
          return { k: 'jump', op: t.v, at };
      }
      if (this.startsDeclaration()) return this.declaration();
    }
    return this.exprStatement();
  }
  private exprStatement(): Stmt {
    const at = this.at();
    const e = this.expression();
    this.expect(';');
    return { k: 'expr', e, at };
  }
  private declaration(): Stmt {
    const at = this.at();
    let isConst = false;
    while (this.peek()?.k === 'id' && QUALIFIERS.has(this.peek()!.v)) isConst = this.name() === 'const' || isConst;
    const type = this.name();
    let arrayType = false;
    if (this.eat('[')) {
      arrayType = true;
      while (!this.eat(']')) this.i++;
    }
    const vars: { name: string; array: boolean; init?: Expr }[] = [];
    do {
      const name = this.name();
      let array = arrayType;
      if (this.eat('[')) {
        array = true;
        while (!this.eat(']')) this.i++;
      }
      vars.push({ name, array, init: this.eat('=') ? this.expression(2) : undefined });
    } while (this.eat(','));
    this.expect(';');
    return { k: 'decl', type, isConst, vars, at };
  }
  /** An expression whose operators bind at least `min` (1: the comma, 2: assignments, 3: the conditional). */
  expression(min = 1): Expr {
    let left = this.unary();
    for (;;) {
      const t = this.peek();
      if (!t || t.k !== 'op') return left;
      const at = t.at;
      if (t.v === ',' && min <= 1) {
        this.i++;
        left = { k: 'seq', a: left, b: this.expression(2), at };
      } else if (ASSIGN.has(t.v) && min <= 2) {
        this.i++;
        left = { k: 'assign', op: t.v, target: left, e: this.expression(2), at };
      } else if (t.v === '?' && min <= 3) {
        this.i++;
        const a = this.expression(2);
        this.expect(':');
        left = { k: 'cond', c: left, a, b: this.expression(2), at };
      } else {
        const power = BINARY_POWER[t.v];
        if (power === undefined || power < min) return left;
        this.i++;
        left = { k: 'binary', op: t.v, a: left, b: this.expression(power + 1), at };
      }
    }
  }
  private unary(): Expr {
    const t = this.peek();
    const at = this.at();
    if (t?.k === 'op' && ['-', '+', '!', '~'].includes(t.v)) {
      this.i++;
      return { k: 'unary', op: t.v, e: this.unary(), at };
    }
    if (t?.k === 'op' && (t.v === '++' || t.v === '--')) {
      this.i++;
      return { k: 'update', op: t.v, prefix: true, e: this.unary(), at };
    }
    let e = this.primary();
    for (;;) {
      const p = this.peek();
      if (p?.v === '.') {
        this.i++;
        e = { k: 'dot', e, name: this.name(), at: p.at };
      } else if (p?.v === '[') {
        this.i++;
        const i = this.expression();
        this.expect(']');
        e = { k: 'index', e, i, at: p.at };
      } else if (p?.v === '++' || p?.v === '--') {
        this.i++;
        e = { k: 'update', op: p.v, prefix: false, e, at: p.at };
      } else return e;
    }
  }
  private primary(): Expr {
    const t = this.peek();
    const at = this.at();
    if (!t) fail('it does not parse: an expression expected at the end');
    this.i++;
    if (t!.k === 'num') {
      const { value, type } = literal(t!.v);
      return { k: 'lit', value, type, at };
    }
    if (t!.v === '(') {
      const e = this.expression();
      this.expect(')');
      return e;
    }
    if (t!.k !== 'id') fail(`it does not parse: "${t!.v}" where an expression starts`);
    if (t!.v === 'true' || t!.v === 'false') return { k: 'lit', value: t!.v === 'true', type: 'bool', at };
    if (this.peek()?.v === '[' && this.isType(t!.v)) {
      // an array constructor, T[n](…) or T[](…) (§5.4.4): what a constant array is initialised with
      this.i++;
      let size: number | undefined;
      if (!this.eat(']')) {
        const n = this.peek();
        if (n?.k !== 'num' || !/^\d+$/.test(n.v)) fail('it sizes an array by other than an int literal');
        this.i++;
        size = Number(n!.v);
        this.expect(']');
      }
      this.expect('(');
      const args: Expr[] = [];
      if (!this.eat(')')) {
        do args.push(this.expression(2));
        while (this.eat(','));
        this.expect(')');
      }
      return { k: 'array', type: t!.v, size, args, at };
    }
    if (this.eat('(')) {
      const args: Expr[] = [];
      if (!this.eat(')')) {
        do args.push(this.expression(2));
        while (this.eat(','));
        this.expect(')');
      }
      return { k: 'call', name: t!.v, args, at };
    }
    return { k: 'id', name: t!.v, at };
  }
}

// ------------------------------------------------------------------------------------------------ programs

interface Param {
  qual: string;
  type: string;
  name: string;
}
interface Def {
  ret: string;
  name: string;
  params: Param[];
  /** The body between its braces, and where it starts in its source. */
  body: string;
  bodyAt: number;
  /** Offsets of the definition (header) and of its end in its source. */
  at: number;
  end: number;
  text: string;
}
interface Const {
  type: string;
  init: string;
  /** The length of a constant array, NaN when it is not an int literal; undefined for a scalar or a vector. */
  size?: number;
}
interface Program {
  defs: Map<string, Def | null>;
  /** Each source's definitions, by position (a block mirror's function is the one around its annotation). */
  sourceDefs: Def[][];
  consts: Map<string, Const | null>;
  structs: Map<string, Map<string, string> | null>;
  /** Uniforms and shader inputs/outputs: names a body may not read. */
  globals: Map<string, string>;
  /** The sampler uniforms, by name, with their types: what texelFetch reads through a binding. */
  samplers: Map<string, string>;
  macros: Map<string, Tok[] | null>;
}

/** Comments as spaces, so that offsets in the text stay those of the source. */
const blankComments = (s: string): string =>
  s.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, (c) => c.replace(/[^\n]/g, ' '));
const blankLine = (line: string): string => line.replace(/[^\n]/g, ' ');

/** One name defined in several sources: kept when every definition is the same, null when two differ. */
function put<T>(map: Map<string, T | null>, key: string, value: T, same: (a: T, b: T) => boolean): void {
  const old = map.get(key);
  map.set(key, old === undefined || (old !== null && same(old, value)) ? value : null);
}

/**
 * The sources compiled together, preprocessed (#ifdef/#ifndef/#else/#endif against the names some source #defines, the
 * directives blanked), with their functions, constants, structs, shader inputs and object-like macros.
 */
function parseProgram(raw: readonly string[]): Program {
  const defined = new Set<string>();
  for (const s of raw) for (const m of blankComments(s).matchAll(/^[ \t]*#[ \t]*define[ \t]+(\w+)/gm)) defined.add(m[1]!);
  // object-like macros; a function-like one (none in the shaders) is never expanded and reads as undefined
  const macroText = new Map<string, string | null>();
  const sources = raw.map((s) => {
    const active: boolean[] = [];
    return blankComments(s)
      .split('\n')
      .map((line) => {
        const d = /^[ \t]*#[ \t]*(\w+)[ \t]*(.*)$/.exec(line);
        const on = active.every(Boolean);
        if (!d) return on ? line : blankLine(line);
        const dir = d[1]!,
          rest = d[2]!;
        if (dir === 'ifdef' || dir === 'ifndef') active.push(defined.has(rest.trim()) === (dir === 'ifdef'));
        else if (dir === 'else') active.push(!active.pop());
        else if (dir === 'endif') active.pop();
        else if (dir === 'if' || dir === 'elif') throw new Error(`shaderMirrors: #${dir} is not supported (${line.trim()})`);
        else if (dir === 'define' && on) {
          const m = /^(\w+)(\()?[ \t]*(.*)$/.exec(rest)!;
          if (!m[2]) put(macroText, m[1]!, m[3]!.trim(), (a, b) => a === b);
        }
        return blankLine(line);
      })
      .join('\n');
  });
  const macros = new Map<string, Tok[] | null>();
  for (const [name, body] of macroText) macros.set(name, body === null ? null : lex(body, 0, new Map()));

  const program: Program = {
    defs: new Map(),
    sourceDefs: [],
    consts: new Map(),
    structs: new Map(),
    globals: new Map(),
    samplers: new Map(),
    macros,
  };
  const structNames = new Set<string>();
  for (const s of sources) for (const m of s.matchAll(/\bstruct\s+(\w+)\s*\{/g)) structNames.add(m[1]!);
  const types = ['void', 'float', 'int', 'uint', 'bool', '[iub]?vec[234]', 'mat[234](?:x[234])?', ...structNames].join('|');
  const header = new RegExp(
    `\\b(?:(?:highp|mediump|lowp|precise)\\s+)?(${types})\\s+(\\w+)\\s*\\(([^()]*)\\)\\s*\\{`,
    'g',
  );
  for (const src of sources) {
    const defs: Def[] = [];
    let outside = '';
    let last = 0;
    header.lastIndex = 0;
    for (let m = header.exec(src); m; m = header.exec(src)) {
      let i = header.lastIndex;
      for (let depth = 1; depth > 0 && i < src.length; i++) depth += src[i] === '{' ? 1 : src[i] === '}' ? -1 : 0;
      const def: Def = {
        ret: m[1]!,
        name: m[2]!,
        params: parseParams(m[3]!),
        body: src.slice(header.lastIndex, i - 1),
        bodyAt: header.lastIndex,
        at: m.index,
        end: i,
        text: src.slice(m.index, i).replace(/\s+/g, ' '),
      };
      defs.push(def);
      put(program.defs, def.name, def, (a, b) => a.text === b.text);
      outside += src.slice(last, m.index);
      last = header.lastIndex = i;
    }
    outside += src.slice(last);
    program.sourceDefs.push(defs);
    for (const m of outside.matchAll(/\bconst\s+(?:(?:high|medium|low)p\s+)?(\w+)\s+(\w+)\s*(\[[^\]]*\])?\s*=\s*([^;]+);/g))
      put(
        program.consts,
        m[2]!,
        { type: m[1]!, init: m[4]!.trim(), size: m[3] ? Number(/^\[\s*(\d+)\s*\]$/.exec(m[3])?.[1] ?? NaN) : undefined },
        (a, b) => a.type === b.type && a.init === b.init && Object.is(a.size, b.size),
      );
    for (const m of outside.matchAll(/\bstruct\s+(\w+)\s*\{([^}]*)\}/g)) {
      const fields = new Map<string, string>();
      for (const f of m[2]!.matchAll(/(?:(?:high|medium|low)p\s+)?(\w+)\s+(\w+)\s*;/g)) fields.set(f[2]!, f[1]!);
      put(program.structs, m[1]!, fields, (a, b) => [...a].join() === [...b].join());
    }
    for (const m of outside.matchAll(/\b(uniform|in|out)\s+(?:(?:high|medium|low)p\s+)?(\w+)\s+(\w+)/g)) {
      program.globals.set(m[3]!, `the ${m[1] === 'uniform' ? 'uniform' : m[1] === 'in' ? 'shader input' : 'shader output'} ${m[3]}`);
      if (m[1] === 'uniform' && /sampler/.test(m[2]!)) program.samplers.set(m[3]!, m[2]!);
    }
  }
  return program;
}

/** `[qualifiers] type name[[n]]`, comma-separated; an array parameter keeps `[]` on its name, which no value takes. */
function parseParams(text: string): Param[] {
  const out: Param[] = [];
  for (const raw of text.split(',')) {
    const words = raw.replace(/\[[^\]]*\]/g, ' [] ').trim().split(/\s+/).filter(Boolean);
    const array = words.includes('[]');
    const plain = words.filter((w) => w !== '[]');
    if (!plain.length || (plain.length === 1 && plain[0] === 'void')) continue;
    let qual = 'in';
    while (plain.length > 2) {
      const w = plain.shift()!;
      if (w === 'out' || w === 'inout' || w === 'in') qual = w;
    }
    out.push({ qual, type: plain[0] ?? '', name: `${plain[1] ?? ''}${array ? '[]' : ''}` });
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ emission

interface Sym {
  type: string;
  js: string;
  mutable: boolean;
}
/** An emitted expression: its JavaScript, its GLSL type, whether it can be assigned and whether reading it twice is safe. */
interface R {
  js: string;
  t: string;
  lvalue?: boolean;
  pure?: boolean;
  /** A swizzle of several components: the vector and the indices, for an assignment through it. */
  swz?: { obj: string; idx: number[] };
}

/** What emitting a function or block found it needs. */
interface Needs {
  callees: Set<string>;
  consts: Set<string>;
  samplers: Set<string>;
  literals: Set<number>;
  parameterReads: Set<number>;
}

const newNeeds = (): Needs => ({
  callees: new Set(),
  consts: new Set(),
  samplers: new Set(),
  literals: new Set(),
  parameterReads: new Set(),
});

/** Emits one function body or block in its scope, typed. */
class Emitter {
  private scopes: Map<string, Sym>[] = [new Map()];
  readonly needs = newNeeds();
  /** A block mirror: the variables of its function, which it reads as inputs. */
  inputs?: Map<string, Sym>;
  readonly used = new Map<string, Sym>();
  /** The function's out and inout parameters other than structs, which it returns after its value. */
  readonly outs: Sym[] = [];
  /** The temporaries of its calls through out parameters, one per call. */
  readonly temps: string[] = [];
  /** A run of statements: the depth of its own scope, whose declarations its function reads after it. */
  private hoistScope?: number;
  readonly hoisted: [string, Sym][] = [];
  constructor(
    private readonly prog: Program,
    private readonly retType: string,
    private readonly blockReturn = false,
  ) {}

  declare(name: string, type: string, mutable: boolean): Sym {
    if (name.endsWith('[]')) fail('it uses arrays');
    checkValueType(type, name, this.prog.structs);
    const sym = { type, js: js(name), mutable };
    this.scopes[this.scopes.length - 1]!.set(name, sym);
    return sym;
  }
  private lookup(name: string): Sym | undefined {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const s = this.scopes[i]!.get(name);
      if (s) return s;
    }
    const input = this.inputs?.get(name);
    if (input) this.used.set(name, input);
    return input;
  }
  private isStruct(t: string): boolean {
    return !!this.prog.structs.get(t);
  }

  statements(list: Stmt[]): string {
    return list.map((s) => this.statement(s)).join('\n');
  }
  /** A run of statements in a scope of its own, whose declarations are left to the caller (`hoisted`). */
  run(list: Stmt[]): string {
    return this.scoped(() => {
      this.hoistScope = this.scopes.length - 1;
      try {
        return this.statements(list);
      } finally {
        this.hoistScope = undefined;
      }
    });
  }
  private scoped(f: () => string): string {
    this.scopes.push(new Map());
    try {
      return f();
    } finally {
      this.scopes.pop();
    }
  }
  private bool(e: Expr, what: string): string {
    const r = this.expr(e);
    if (r.t !== 'bool') fail(`${what} is ${r.t}, not bool`);
    return r.js;
  }
  statement(s: Stmt): string {
    switch (s.k) {
      case 'decl': {
        if (s.vars.some((v) => v.array)) fail('it uses arrays');
        const hoist = this.hoistScope === this.scopes.length - 1;
        const parts = s.vars.map((v) => {
          let init: string;
          if (v.init) {
            const r = this.expr(v.init);
            if (r.t !== s.type) fail(`it initialises the ${s.type} ${v.name} with a ${r.t}`);
            init = isVec(s.type) ? `__copy(${r.js})` : this.isStruct(s.type) ? `__clone(${r.js})` : r.js;
          } else init = undefinedOf(s.type, this.prog.structs);
          const sym = this.declare(v.name, s.type, !s.isConst);
          if (hoist) this.hoisted.push([v.name, sym]);
          return `${sym.js} = ${init}`;
        });
        return hoist ? `${parts.join(', ')};` : `${s.isConst ? 'const' : 'let'} ${parts.join(', ')};`;
      }
      case 'expr':
        return `${this.expr(s.e).js};`;
      case 'block':
        return this.scoped(() => `{\n${this.statements(s.body)}\n}`);
      case 'if':
        return `if (${this.bool(s.c, 'a condition')}) ${this.scoped(() => this.statement(s.t))}${
          s.f ? ` else ${this.scoped(() => this.statement(s.f!))}` : ''
        }`;
      case 'for':
        return this.scoped(() => {
          const init = s.init ? this.statement(s.init) : ';';
          const c = s.c ? this.bool(s.c, 'a loop condition') : '';
          const step = s.step ? this.expr(s.step).js : '';
          return `for (${init} ${c}; ${step}) ${this.scoped(() => this.statement(s.body))}`;
        });
      case 'while':
        return `while (${this.bool(s.c, 'a loop condition')}) ${this.scoped(() => this.statement(s.body))}`;
      case 'do':
        return `do ${this.scoped(() => this.statement(s.body))} while (${this.bool(s.c, 'a loop condition')});`;
      case 'return': {
        const r = s.e ? this.expr(s.e) : { js: 'undefined', t: 'void' };
        if (r.t !== this.retType) fail(`it returns a ${r.t} from a ${this.retType} function`);
        if (this.blockReturn) return `return { returned: ${r.js}, values: __values() };`;
        return this.outs.length ? `return [${r.js}, ${this.outs.map((o) => o.js).join(', ')}];` : `return ${r.js};`;
      }
      case 'jump':
        if (s.op === 'discard') fail('it discards the fragment');
        return `${s.op};`;
    }
  }

  expr(e: Expr): R {
    switch (e.k) {
      case 'lit':
        if (e.type === 'uint') fail('it uses uint, whose arithmetic wraps modulo 2³² and is not emulated');
        if (typeof e.value === 'number') {
          if (e.type === 'int' && !(e.value <= 2147483647)) fail(`the int literal ${e.value} does not fit 32 bits`);
          this.needs.literals.add(e.value);
        }
        return { js: String(e.value), t: e.type, pure: true };
      case 'id':
        return this.identifier(e.name);
      case 'call':
        return this.call(e.name, e.args);
      case 'dot':
        return this.select(this.expr(e.e), e.name);
      case 'index': {
        const o = this.expr(e.e);
        const i = this.expr(e.i);
        const array = ARRAY.exec(o.t);
        if (array) {
          // an element of a constant array; an index beyond it is undefined (§5.7)
          if (i.t !== 'int') fail(`it indexes an array with a ${i.t}`);
          const n = Number(array[2]);
          const k = /^\d+$/.test(i.js) ? Number(i.js) : undefined;
          if (k !== undefined && k >= n) fail(`it reads element ${k} of a ${o.t}`);
          return { js: `${o.js}[${k ?? `__ix(${n}, ${i.js})`}]`, t: array[1]!, pure: o.pure && i.pure };
        }
        if (!isVec(o.t)) fail('it uses arrays');
        if (i.t !== 'int') fail(`it indexes a vector with a ${i.t}`);
        // an index out of the vector is undefined (§5.5): a constant one is named, a computed one raises GlslUndefined
        const n = o.swz ? o.swz.idx.length : sizeOf(o.t);
        const k = /^\d+$/.test(i.js) ? Number(i.js) : undefined;
        if (k !== undefined && k >= n) fail(`it reads component ${k} of a ${o.swz ? vecOf(componentOf(o.t), n) : o.t}`);
        const at = k ?? `__ix(${n}, ${i.js})`;
        const pure = o.pure && i.pure;
        // an index of a swizzle is the component of the swizzled vector, which an assignment through it writes
        if (o.swz) {
          const c = k !== undefined ? String(o.swz.idx[k]) : `[${o.swz.idx}][${at}]`;
          return { js: `${o.swz.obj}[${c}]`, t: componentOf(o.t), lvalue: o.lvalue, pure };
        }
        return { js: `${o.js}[${at}]`, t: componentOf(o.t), lvalue: o.lvalue, pure };
      }
      case 'unary': {
        const a = this.expr(e.e);
        if (e.op === '!') {
          if (a.t !== 'bool') fail(`it negates a ${a.t} with !`);
          return { js: `(!${a.js})`, t: 'bool', pure: a.pure };
        }
        if (a.t === 'bool' || this.isStruct(a.t) || isArray(a.t)) fail(`it applies ${e.op} to a ${a.t}`);
        if (e.op === '~') {
          if (!isIntType(a.t)) fail(`it applies the bitwise ~ to a ${a.t}`);
          return { js: isVec(a.t) ? `__iv.not(${a.js})` : `(~${a.js})`, t: a.t, pure: a.pure };
        }
        if (e.op === '+') return { js: a.js, t: a.t, pure: a.pure };
        // int negation wraps as the rest of the int arithmetic does: −(−2³¹) is −2³¹
        if (isIntType(a.t)) return { js: isVec(a.t) ? `__iv.neg(${a.js})` : `((- ${a.js}) | 0)`, t: a.t, pure: a.pure };
        return { js: isVec(a.t) ? `__neg(${a.js})` : `(- ${a.js})`, t: a.t, pure: a.pure };
      }
      case 'update': {
        const a = this.expr(e.e);
        if (!a.lvalue || a.swz) fail('it increments something that is not a variable');
        if (a.t === 'float') return { js: e.prefix ? `(${e.op}${a.js})` : `(${a.js}${e.op})`, t: a.t };
        if (a.t !== 'int') fail(`it increments a ${a.t}`);
        if (!a.pure) fail('it increments an element whose index has side effects');
        const set = `(${a.js} = (${a.js} ${e.op[0]} 1) | 0)`;
        return { js: e.prefix ? set : `((${set} ${e.op[0] === '+' ? '-' : '+'} 1) | 0)`, t: 'int' };
      }
      case 'binary':
        return this.binary(e.op, this.expr(e.a), this.expr(e.b));
      case 'cond': {
        const c = this.bool(e.c, 'a condition');
        const a = this.expr(e.a),
          b = this.expr(e.b);
        if (a.t !== b.t) fail(`its ?: chooses between a ${a.t} and a ${b.t}`);
        return { js: `(${c} ? ${a.js} : ${b.js})`, t: a.t };
      }
      case 'assign': {
        const target = this.expr(e.target);
        if (!target.lvalue) fail('it assigns to something that is not a variable');
        let value: R;
        if (e.op === '=') value = this.expr(e.e);
        else {
          if (!target.pure) fail('it updates an element whose index has side effects');
          value = this.binary(e.op.slice(0, -1), target, this.expr(e.e));
        }
        if (value.t !== target.t) fail(`it assigns a ${value.t} to a ${target.t}`);
        return { js: this.store(target, value.js), t: target.t };
      }
      case 'seq': {
        const a = this.expr(e.a),
          b = this.expr(e.b);
        return { js: `(${a.js}, ${b.js})`, t: b.t };
      }
      case 'array': {
        // the initialiser of a constant array of scalars: its elements of its type exactly (no conversion, §5.4.4)
        if (!/^(?:float|int|bool)$/.test(e.type)) fail(`it builds an array of ${e.type}`);
        const parts = e.args.map((a) => this.expr(a));
        const bad = parts.find((p) => p.t !== e.type);
        if (bad) fail(`it builds a ${e.type} array from a ${bad.t}`);
        const n = e.size ?? parts.length;
        if (parts.length !== n || n < 1) fail(`it builds a ${e.type}[${n}] from ${parts.length} elements`);
        return { js: `[${parts.map((p) => p.js).join(', ')}]`, t: `${e.type}[${n}]`, pure: parts.every((p) => p.pure) };
      }
    }
  }

  /** An assignment to an l-value, which receives a copy of a vector or struct as GLSL assigns them. */
  private store(target: R, value: string): string {
    if (this.isStruct(target.t)) {
      if (!target.pure) fail('it assigns a struct to an element whose index has side effects');
      return `(${target.js} = __assignStruct(${target.js}, ${value}))`;
    }
    const v = isVec(target.t) ? `__copy(${value})` : value;
    if (target.swz) return `__swzSet(${target.swz.obj}, [${target.swz.idx}], ${v})`;
    return `(${target.js} = ${v})`;
  }

  private identifier(name: string): R {
    const s = this.lookup(name);
    if (s) return { js: s.js, t: s.type, lvalue: s.mutable, pure: true };
    if (this.prog.macros.get(name) === null) fail(`${name} is defined differently in two places`);
    const c = this.prog.consts.get(name);
    if (c === null) fail(`${name} is defined differently in two places`);
    if (c) {
      if (c.size !== undefined && !(c.size >= 1)) fail(`it reads the array ${name}, sized otherwise than by an int literal`);
      this.needs.consts.add(name);
      return { js: js(name), t: c.size === undefined ? c.type : `${c.type}[${c.size}]`, pure: true };
    }
    const g = this.prog.globals.get(name);
    if (g) fail(`it reads ${g}`);
    return fail(`${name} is not defined`);
  }

  private select(o: R, name: string): R {
    if (isVec(o.t)) {
      const set = SWIZZLE_SETS.find((s) => [...name].every((c) => s.includes(c)));
      const idx = set ? [...name].map((c) => set.indexOf(c)) : [];
      if (!set || name.length > 4 || idx.some((i) => i >= sizeOf(o.t))) fail(`it selects .${name} of a ${o.t}`);
      const component = componentOf(o.t);
      // a swizzle of a swizzle selects from the vector beneath (§5.5): v.zyx.x is v.z, and an assignment through it
      // writes v, not a copy of v.zyx
      const swz = o.swz ? { obj: o.swz.obj, idx: idx.map((i) => o.swz!.idx[i]!) } : { obj: o.js, idx };
      if (idx.length === 1) return { js: `${swz.obj}[${swz.idx[0]}]`, t: component, lvalue: o.lvalue, pure: o.pure };
      const distinct = new Set(idx).size === idx.length;
      return {
        js: `__swz(${swz.obj}, [${swz.idx}])`,
        t: vecOf(component, idx.length),
        lvalue: o.lvalue && distinct,
        pure: o.pure,
        swz,
      };
    }
    const fields = this.prog.structs.get(o.t);
    if (fields) {
      const t = fields.get(name);
      if (!t) fail(`the struct ${o.t} has no field ${name}`);
      return { js: `${o.js}.${name}`, t: t!, lvalue: o.lvalue, pure: o.pure };
    }
    return fail(`it selects .${name} of a ${o.t}`);
  }

  binary(op: string, a: R, b: R): R {
    if (isArray(a.t) || isArray(b.t)) fail(`it applies ${op} to a ${a.t} and a ${b.t}`);
    if (op === '&&' || op === '||' || op === '^^') {
      if (a.t !== 'bool' || b.t !== 'bool') fail(`it applies ${op} to a ${a.t} and a ${b.t}`);
      return { js: op === '^^' ? `(${a.js} !== ${b.js})` : `(${a.js} ${op} ${b.js})`, t: 'bool' };
    }
    if (op === '==' || op === '!=') {
      if (a.t !== b.t) fail(`it compares a ${a.t} with a ${b.t}`);
      if (isVec(a.t)) return { js: `${op === '!=' ? '!' : ''}__eq(${a.js}, ${b.js})`, t: 'bool' };
      if (this.isStruct(a.t)) fail('it compares structs');
      return { js: `(${a.js} ${op}= ${b.js})`, t: 'bool' };
    }
    if (a.t === 'bool' || b.t === 'bool' || this.isStruct(a.t) || this.isStruct(b.t)) fail(`it applies ${op} to a ${a.t} and a ${b.t}`);
    if (['<', '>', '<=', '>='].includes(op)) {
      if (a.t !== b.t || isVec(a.t)) fail(`it compares a ${a.t} with a ${b.t} by ${op}`);
      return { js: `(${a.js} ${op} ${b.js})`, t: 'bool' };
    }
    const bitwise = ['&', '|', '^', '<<', '>>'].includes(op);
    if (bitwise && (!isIntType(a.t) || !isIntType(b.t))) fail(`it applies the bitwise ${op} to a ${a.t} and a ${b.t}`);
    if (isVec(a.t) && isVec(b.t) && a.t !== b.t) fail(`it combines a ${a.t} and a ${b.t}`);
    if (componentOf(a.t) !== componentOf(b.t)) fail(`it combines a ${a.t} and a ${b.t}, which GLSL ES 3.00 does not convert`);
    if (isIntType(a.t)) {
      // a shift takes its type from the left operand, and a vector count needs a vector to shift (§5.9)
      if ((op === '<<' || op === '>>') && !isVec(a.t) && isVec(b.t)) fail(`it shifts an int by a ${b.t}`);
      if (isVec(a.t) || isVec(b.t)) return { js: `__iv[${JSON.stringify(op)}](${a.js}, ${b.js})`, t: isVec(a.t) ? a.t : b.t };
      switch (op) {
        case '+':
        case '-':
          return { js: `((${a.js} ${op} ${b.js}) | 0)`, t: 'int' };
        case '*':
          return { js: `Math.imul(${a.js}, ${b.js})`, t: 'int' };
        case '/':
          return { js: `__idiv(${a.js}, ${b.js})`, t: 'int' };
        case '%':
          return { js: `__imod(${a.js}, ${b.js})`, t: 'int' };
        case '<<':
          return { js: `__shl(${a.js}, ${b.js})`, t: 'int' };
        case '>>':
          return { js: `__shr(${a.js}, ${b.js})`, t: 'int' };
        default:
          return { js: `(${a.js} ${op} ${b.js})`, t: 'int' };
      }
    }
    if (op === '%') fail(`it applies % to a ${a.t} and a ${b.t}`);
    if (isVec(a.t) || isVec(b.t)) {
      const f = { '+': '__add', '-': '__sub', '*': '__mul', '/': '__div' }[op]!;
      return { js: `${f}(${a.js}, ${b.js})`, t: isVec(a.t) ? a.t : b.t };
    }
    return { js: `(${a.js} ${op} ${b.js})`, t: a.t };
  }

  private call(name: string, argExprs: Expr[]): R {
    if (name === 'uint' || /^uvec/.test(name)) fail('it uses uint, whose arithmetic wraps modulo 2³² and is not emulated');
    if (/^bvec[234]$/.test(name)) fail(`it constructs a boolean vector (${name})`);
    if (/^mat/.test(name)) fail(`it constructs a matrix (${name})`);
    if (name === 'texelFetch') return this.texelFetch(argExprs);
    if (/^(?:texture\w*|texelFetchOffset)$/.test(name) && !this.prog.defs.has(name))
      fail(`it samples a texture through ${name}, whose filtering and addressing are not emulated`);
    const args = argExprs.map((a) => this.expr(a));
    if (name === 'float' || name === 'int' || name === 'bool') return this.convert(name, args);
    if (isVec(name)) return this.construct(name, args);
    const d = this.prog.defs.get(name);
    if (d === null) fail(`${name} is defined differently in two places`);
    if (d) return this.userCall(d, args);
    if (name === 'round') fail('it uses round, whose rounding of halves GLSL ES 3.00 leaves to the implementation');
    if (name in BUILTINS || ['length', 'distance', 'dot', 'cross', 'normalize'].includes(name)) return this.builtin(name, args);
    return fail(`${name} is neither a GLSL built-in nor a function of the program`);
  }

  /** texelFetch(sampler, coordinates, level): the texel the binding of the sampler reads from the uploaded data. */
  private texelFetch(argExprs: Expr[]): R {
    const [sampler, ...rest] = argExprs;
    if (sampler?.k !== 'id' || rest.length !== 2) fail('it calls texelFetch with other than a sampler, its coordinates and a level');
    const s = sampler as { name: string };
    const type = this.prog.samplers.get(s.name);
    if (!type) fail(`it reads ${s.name} as a texture, which is no sampler uniform of the program`);
    const m = /^(i|u)?sampler([23])D$/.exec(type!);
    if (!m) fail(`it reads the ${type} ${s.name}`);
    if (m![1] === 'u') fail(`it reads the ${type} ${s.name}, whose texels are uint`);
    const [coord, lod] = rest.map((a) => this.expr(a)) as [R, R];
    if (coord.t !== `ivec${m![2]}` || lod.t !== 'int') fail(`it reads ${s.name} at a ${coord.t} and a level ${lod.t}`);
    this.needs.samplers.add(s.name);
    return { js: `__tex_${s.name}(${coord.js}, ${lod.js})`, t: m![1] === 'i' ? 'ivec4' : 'vec4' };
  }

  private convert(to: string, args: R[]): R {
    if (args.length !== 1) fail(`${to}() takes one argument`);
    const a = args[0]!;
    if (this.isStruct(a.t) || isArray(a.t)) fail(`it converts a ${a.t} to ${to}`);
    const first = isVec(a.t) ? `${a.js}[0]` : a.js;
    const from = componentOf(a.t);
    const fromBool = `(${first} ? 1 : 0)`;
    if (to === 'float') return { js: from === 'bool' ? fromBool : first, t: 'float' };
    if (to === 'int') return { js: from === 'bool' ? fromBool : from === 'float' ? `__toInt(${first})` : first, t: 'int' };
    return { js: from === 'bool' ? first : `(${first} !== 0)`, t: 'bool' };
  }

  /** A vector constructor: its components converted to the vector's (an int one drops the fraction, §5.4.1). */
  private construct(type: string, args: R[]): R {
    const n = sizeOf(type);
    const toInt = componentOf(type) === 'int';
    const parts = args.map((a) => {
      if (this.isStruct(a.t) || isArray(a.t)) fail(`it builds a vector from a ${a.t}`);
      const from = componentOf(a.t);
      if (from === 'bool') return `(${a.js} ? 1 : 0)`;
      if (toInt && from === 'float') return isVec(a.t) ? `${a.js}.map((x) => __toInt(x))` : `__toInt(${a.js})`;
      return a.js;
    });
    if (args.length === 1 && !isVec(args[0]!.t)) return { js: `__splat(${n}, ${parts[0]})`, t: type };
    const components = args.reduce((s, a) => s + sizeOf(a.t), 0);
    if (components < n) fail(`it builds a ${type} from ${components} components`);
    if (args.every((a) => !isVec(a.t)) && components === n) return { js: `[${parts.join(', ')}]`, t: type };
    return { js: `__cat(${n}, ${parts.join(', ')})`, t: type };
  }

  /**
   * A call of a function of the program. GLSL passes by value: an out or inout argument (other than a struct, which the
   * callee writes in place) receives the parameter's value when the callee returns, which the callee returns after its
   * own value; the call keeps the pair in a temporary of its own.
   */
  private userCall(d: Def, args: R[]): R {
    if (args.length !== d.params.length) fail(`it calls ${d.name} with ${args.length} arguments, not ${d.params.length}`);
    d.params.forEach((p, i) => {
      const a = args[i]!;
      if (a.t !== p.type) fail(`it passes a ${a.t} to the ${p.type} ${p.name} of ${d.name}`);
      if (p.qual !== 'in' && !a.lvalue) fail(`it passes to the ${p.qual} ${p.name} of ${d.name} what is not a variable`);
    });
    this.needs.callees.add(d.name);
    const lit = d.name === 'P' && /^\d+$/.test(args[0]?.js ?? '') ? Number(args[0]!.js) : undefined;
    if (lit !== undefined) this.needs.parameterReads.add(lit);
    const call = `${js(d.name)}(${args.map((a) => a.js).join(', ')})`;
    const outs = d.params.flatMap((p, i) => (p.qual !== 'in' && !this.isStruct(p.type) ? [i] : []));
    if (!outs.length) return { js: call, t: d.ret };
    for (const i of outs) if (!args[i]!.pure) fail(`it passes to the ${d.params[i]!.qual} ${d.params[i]!.name} of ${d.name} an element whose index has side effects`);
    const tmp = `__o${this.temps.length}`;
    this.temps.push(tmp);
    const stores = outs.map((i, k) => this.store(args[i]!, `${tmp}[${k + 1}]`));
    return { js: `(${tmp} = ${call}, ${stores.join(', ')}, ${tmp}[0])`, t: d.ret };
  }

  private builtin(name: string, args: R[]): R {
    const types = args.map((a) => a.t);
    const list = args.map((a) => a.js).join(', ');
    const vec = types.find(isVec);
    const genType = (t: string) => t === 'float' || (isVec(t) && componentOf(t) === 'float');
    const same = (t: string) => types.every((u) => u === t);
    const arity = (n: number[]) => {
      if (!n.includes(args.length)) fail(`it calls ${name} with ${args.length} arguments`);
    };
    switch (name) {
      case 'length':
      case 'normalize':
        arity([1]);
        if (!genType(types[0]!)) fail(`it calls ${name} on a ${types[0]}`);
        return { js: `__${name}(${list})`, t: name === 'length' ? 'float' : types[0]! };
      case 'distance':
      case 'dot':
        arity([2]);
        if (!genType(types[0]!) || !same(types[0]!)) fail(`it calls ${name} on a ${types.join(' and a ')}`);
        if (name === 'dot') return { js: `__dot(${list})`, t: 'float' };
        return { js: vec ? `__length(__sub(${list}))` : `__length(${args[0]!.js} - ${args[1]!.js})`, t: 'float' };
      case 'cross':
        arity([2]);
        if (!same('vec3')) fail(`it calls cross on a ${types.join(' and a ')}`);
        return { js: `__cross(${list})`, t: 'vec3' };
      case 'isnan':
      case 'isinf':
        arity([1]);
        if (types[0] !== 'float') fail(`it calls ${name} on a ${types[0]}`);
        return { js: `${name}(${list})`, t: 'bool' };
    }
    // the arguments that may be scalar where the others are vectors (§8): the x of a genType function sets the type
    const scalarOk: Record<string, number[]> = { min: [1], max: [1], mod: [1], clamp: [1, 2], mix: [2], step: [0], smoothstep: [0, 1] };
    const fits = (t: string, component: string) =>
      types.forEach((u, i) => {
        if (u !== t && !(u === component && scalarOk[name]?.includes(i))) fail(`it calls ${name} on a ${types.join(' and a ')}`);
      });
    // abs, sign, min, max and clamp have int forms (§8.3), which these emit as the float ones
    if (INT_BUILTINS.has(name) && types.every(isIntType)) {
      fits(vec ?? 'int', 'int');
      return { js: vec ? `__v.${name}(${list})` : `${name}(${list})`, t: vec ?? 'int' };
    }
    if (!types.every(genType)) fail(`it calls ${name} on a ${types.join(' and a ')}, which GLSL ES 3.00 does not convert`);
    const t = vec ?? 'float';
    fits(t, 'float');
    if (name === 'atan') arity([1, 2]);
    return { js: vec ? `__v.${name}(${list})` : `${name}(${list})`, t };
  }
}

// ------------------------------------------------------------------------------------------------ mirrors

/** One annotation and the offset, in its source, of the definition, block or statement it is followed by. */
interface Annotation {
  tsPath: string;
  tsName: string;
  source: number;
  at: number;
}

function annotationsOf(raw: string, source: number): Annotation[] {
  const out: Annotation[] = [];
  for (const m of raw.matchAll(MIRROR_ANNOTATION)) {
    const eol = raw.indexOf('\n', m.index);
    // past the rest of its line, blank lines and comments, to the definition, block or statement it names
    const skip = /(?:\s+|\/\/[^\n]*|\/\*[\s\S]*?\*\/)*/y;
    skip.lastIndex = eol < 0 ? raw.length : eol;
    skip.exec(raw);
    out.push({ tsPath: m[1]!, tsName: m[2]!, source, at: skip.lastIndex });
  }
  return out;
}

/** The line that closes a run of statements annotated as the mirror of `name`. */
const runEnd = (name: string): RegExp => new RegExp(`//[ \\t]*end of ${name}\\b`);

/**
 * The statements an annotation inside a function names, with the declarations visible before them (parameters first):
 * the block `{ … }` at `at`, or, given its `end`, the run of statements of one list from `at` to there. Why there are
 * none, when there are none.
 */
function statementsAt(stmts: Stmt[], params: Param[], at: number, end?: number): { list: Stmt[]; visible: [string, string][] } | string {
  const visible: [string, string][] = params.map((p) => [p.name, p.type]);
  let found: Stmt[] | string | undefined;
  const walk = (list: Stmt[]): boolean => {
    for (let i = 0; i < list.length; i++) {
      const s = list[i]!;
      if (s.at === at && (end !== undefined || s.k === 'block')) {
        let j = i + 1;
        while (end !== undefined && j < list.length && list[j]!.at < end) j++;
        const last = list[j - 1]!;
        found = end !== undefined && (last.end ?? last.at) > end ? 'its end lies inside one of its statements' : list.slice(i, j);
        return true;
      }
      const inner: Stmt[] =
        s.k === 'block'
          ? s.body
          : s.k === 'if'
            ? [s.t, ...(s.f ? [s.f] : [])]
            : s.k === 'for'
              ? [...(s.init ? [s.init] : []), s.body]
              : s.k === 'while' || s.k === 'do'
                ? [s.body]
                : [];
      const mark = visible.length;
      if (inner.length && walk(inner)) return true;
      visible.length = mark;
      if (s.k === 'decl') for (const v of s.vars) visible.push([v.name, s.type]);
    }
    return false;
  };
  walk(stmts);
  if (found === undefined)
    return end === undefined ? 'the annotated block is not one of its function’s statements' : 'the annotated statements are not statements of its function';
  return typeof found === 'string' ? found : { list: found, visible };
}

/** The annotated mirrors of one program (the sources compiled together), translated with that program's names. */
function programMirrors(raw: readonly string[]): GlslMirror[] {
  const annotations = raw.flatMap((s, i) => annotationsOf(s, i));
  if (!annotations.length) return [];
  const prog = parseProgram(raw);
  const isType = (n: string) => BUILTIN_TYPES.test(n) || prog.structs.has(n);
  const parse = (d: Def): Stmt[] => new Parser(lex(d.body, d.bodyAt, prog.macros), isType).statements();
  const isStruct = (t: string) => !!prog.structs.get(t);

  // each function's translation, or why it has none, with what it needs
  const emitted = new Map<string, { js?: string; reason?: string; needs: Needs }>();
  const emitFunction = (name: string) => {
    let r = emitted.get(name);
    if (r) return r;
    const d = prog.defs.get(name)!;
    const em = new Emitter(prog, d.ret);
    try {
      checkValueType(d.ret, 'its result', prog.structs);
      if (isStruct(d.ret)) fail(`it returns a struct ${d.ret}`);
      // GLSL passes by value (§6.1.1): in copies the argument, out starts undefined, inout copies it in, and out and
      // inout copy the parameter back on return, which the call does from what the function returns after its value. A
      // struct out or inout parameter is the caller's object itself, written in place, an out one emptied to undefined
      // fields once the in ones are copied: that differs from GLSL's copy back only where one call passes a struct to two
      // out or inout parameters, which no shader can (one struct, Sample, and no function takes two).
      const prologue: string[] = [];
      const outStructs: string[] = [];
      for (const p of d.params) {
        const sym = em.declare(p.name, p.type, true);
        if (isStruct(p.type)) {
          if (p.qual === 'in') prologue.push(`${sym.js} = __clone(${sym.js});`);
          else if (p.qual === 'out') outStructs.push(`__assignStruct(${sym.js}, ${undefinedOf(p.type, prog.structs)});`);
          continue;
        }
        if (p.qual === 'out') prologue.push(`${sym.js} = ${isVec(p.type) ? `__undef(${sizeOf(p.type)})` : 'undefined'};`);
        else if (isVec(p.type)) prologue.push(`${sym.js} = __copy(${sym.js});`);
        if (p.qual !== 'in') em.outs.push(sym);
      }
      prologue.push(...outStructs);
      const body = em.statements(parse(d));
      if (em.temps.length) prologue.unshift(`let ${em.temps.join(', ')};`);
      const tail = em.outs.length ? `\nreturn [undefined, ${em.outs.map((o) => o.js).join(', ')}];` : '';
      r = {
        js: `function ${js(d.name)}(${d.params.map((p) => js(p.name)).join(', ')}) {\n${prologue.join('\n')}\n${body}${tail}\n}`,
        needs: em.needs,
      };
    } catch (e) {
      if (!(e instanceof Untranslatable)) throw e;
      r = { reason: e.message, needs: em.needs };
    }
    emitted.set(name, r);
    return r;
  };
  const emittedConsts = new Map<string, { js?: string; reason?: string; needs: Needs }>();
  const emitConst = (name: string) => {
    let r = emittedConsts.get(name);
    if (r) return r;
    const c = prog.consts.get(name)!;
    const em = new Emitter(prog, c.type);
    try {
      const v = em.expr(new Parser(lex(c.init, 0, prog.macros), isType).expression(2));
      const type = c.size === undefined ? c.type : `${c.type}[${c.size}]`;
      if (v.t !== type) fail(`the ${type} ${name} is initialised with a ${v.t}`);
      if (em.temps.length) fail(`the ${type} ${name} is initialised through out parameters`);
      r = { js: `const ${js(name)} = ${isVec(type) || isArray(type) ? `Object.freeze(${v.js})` : v.js};`, needs: em.needs };
    } catch (e) {
      if (!(e instanceof Untranslatable)) throw e;
      r = { reason: e.message, needs: em.needs };
    }
    emittedConsts.set(name, r);
    return r;
  };

  // the functions and constants a translation reaches, the samplers they read, and the first reason one has none
  const closure = (needs: Needs) => {
    const fns = new Set<string>(),
      consts = new Set<string>(),
      samplers = new Set(needs.samplers);
    const literals = new Set<number>([0, 1, ...needs.literals]);
    const reads = new Set(needs.parameterReads);
    let reason: string | undefined;
    const visit = (n: Needs) => {
      for (const c of n.consts) {
        if (consts.has(c)) continue;
        consts.add(c);
        const r = emitConst(c);
        if (r.reason) reason ??= `${c} is outside the translation: ${r.reason}`;
        visit(r.needs);
      }
      for (const f of n.callees) {
        if (fns.has(f)) continue;
        fns.add(f);
        const r = emitFunction(f);
        if (r.reason) reason ??= `${f} is outside the translation: ${r.reason}`;
        visit(r.needs);
      }
      for (const s of n.samplers) samplers.add(s);
      for (const l of n.literals) literals.add(l);
      for (const p of n.parameterReads) reads.add(p);
    };
    visit(needs);
    return { fns, consts, samplers, literals, reads, reason };
  };

  // one module per mirror, instantiated per binding: each function may be replaced, each sampler must be bound
  const instantiate = (consts: Set<string>, fns: Set<string>, extra: string, samplers: Set<string>, exported: string) => {
    const constLines: string[] = [];
    const placed = new Set<string>();
    const place = (c: string) => {
      if (placed.has(c)) return;
      placed.add(c);
      for (const d of emitConst(c).needs.consts) place(d);
      constLines.push(emitConst(c).js!);
    };
    for (const c of consts) place(c);
    const text = [
      '"use strict";',
      ...constLines,
      ...[...fns].map((f) => emitted.get(f)!.js!.replace(/^function ([\w$]+)/, 'function __glsl_$1')),
      ...[...fns].map((n) => `const ${js(n)} = __bound.${n} ?? __glsl_${js(n)};`),
      ...[...samplers].map((s) => `const __tex_${s} = __bound.${s} ?? __unbound(${JSON.stringify(s)});`),
      extra,
      `return ${exported};`,
    ].join('\n');
    const factory = new Function(...RUNTIME_NAMES, '__bound', '__unbound', text) as (...a: unknown[]) => unknown;
    const unbound = (n: string) => () => {
      throw new Error(`${n} is a texture the translation reads: bind its texels`);
    };
    // a binding named like a scalar built-in replaces it in this instance, for every function the mirror reaches
    return (bindings: GlslBindings) =>
      factory(...RUNTIME_NAMES.map((n, i) => (n in BUILTINS && Object.hasOwn(bindings, n) ? bindings[n] : RUNTIME_VALUES[i])), bindings, unbound);
  };

  const mirrors: GlslMirror[] = [];
  for (const a of annotations) {
    const base = {
      tsPath: a.tsPath,
      tsName: a.tsName,
      inputs: [] as string[],
      outputs: [] as string[],
      reads: [] as string[],
      landmarks: [0, 1],
      parameterReads: [] as number[],
    };
    const defs = prog.sourceDefs[a.source]!;
    const d = defs.find((f) => f.at === a.at);
    if (!d) {
      // a block `{ … }` of a function, or a run of its statements up to `// end of <export>`
      const fn = defs.find((f) => f.bodyAt <= a.at && a.at < f.end);
      if (!fn) {
        mirrors.push({ ...base, kind: 'function', glslName: '', paramTypes: [], reason: 'no function definition follows the annotation' });
        continue;
      }
      const glslName = `${fn.name}{${a.tsName}}`;
      const em = new Emitter(prog, fn.ret, true);
      try {
        const isBlock = raw[a.source]![a.at] === '{';
        let end: number | undefined;
        if (!isBlock) {
          const k = raw[a.source]!.slice(a.at, fn.end).search(runEnd(a.tsName));
          if (k < 0) fail(`no "// end of ${a.tsName}" closes the statements it names`);
          end = a.at + k;
        }
        const found = statementsAt(parse(fn), fn.params, a.at, end);
        if (typeof found === 'string') fail(found);
        const { list, visible } = found as { list: Stmt[]; visible: [string, string][] };
        em.inputs = new Map();
        for (const [name, type] of visible) {
          if (isStruct(type) || isVec(type) || /^(?:float|int|bool)$/.test(type)) em.inputs.set(name, { type, js: js(name), mutable: true });
        }
        const body = isBlock ? em.statement(list[0]!) : em.run(list);
        const used = [...em.used];
        for (const [n] of em.hoisted) if (em.used.has(n)) fail(`it declares ${n}, which it also reads from its function`);
        const declared = em.hoisted;
        // a vector input is copied, as the block's variable it stands for; a struct is the caller's object
        const prologue = [
          ...(declared.length ? [`let ${declared.map(([, s]) => s.js).join(', ')};`] : []),
          ...(em.temps.length ? [`let ${em.temps.join(', ')};`] : []),
          ...used.map(([n, s]) => `let ${s.js} = ${isVec(s.type) ? `__copy(__in.${n})` : `__in.${n}`};`),
        ].join('\n');
        const missing = used.map(([n]) => `if (!(${JSON.stringify(n)} in __in)) throw new Error('the block reads ${n}, which its runner was not given');`);
        const values = `const __values = () => ({ ${[...used, ...declared].map(([n, s]) => `${JSON.stringify(n)}: ${s.js}`).join(', ')} });`;
        const extra = `function __block(__in) {\n${missing.join('\n')}\n${prologue}\n${values}\n${body}\nreturn { returned: undefined, values: __values() };\n}`;
        const c = closure(em.needs);
        if (c.reason) fail(c.reason);
        const make = instantiate(c.consts, c.fns, extra, c.samplers, '__block');
        mirrors.push({
          ...base,
          kind: 'block',
          glslName,
          inputs: used.map(([n]) => n),
          outputs: declared.map(([n]) => n),
          paramTypes: used.map(([, s]) => s.type),
          reads: [...c.samplers],
          bindBlock: (bindings) => make(bindings) as (inputs: Record<string, unknown>) => GlslBlockRun,
          landmarks: [...c.literals],
          parameterReads: [...c.reads],
        });
      } catch (e) {
        if (!(e instanceof Untranslatable)) throw e;
        mirrors.push({ ...base, kind: 'block', glslName, paramTypes: [], reason: e.message });
      }
      continue;
    }
    const glslName = d.name;
    const paramTypes = d.params.map((p) => p.type);
    if (prog.defs.get(glslName) === null) {
      mirrors.push({ ...base, kind: 'function', glslName, paramTypes, reason: `${glslName} is defined differently in two places` });
      continue;
    }
    const own = emitFunction(glslName);
    const c = closure({ ...newNeeds(), callees: new Set([glslName]) });
    const reason = own.reason ? `${glslName} is outside the translation: ${own.reason}` : c.reason;
    if (reason) {
      mirrors.push({ ...base, kind: 'function', glslName, paramTypes, reads: [...c.samplers], reason, landmarks: [...c.literals] });
      continue;
    }
    const make = instantiate(c.consts, c.fns, '', c.samplers, js(glslName));
    const bind = (bindings: GlslBindings) => make(bindings) as (...args: GlslValue[]) => GlslValue;
    mirrors.push({
      ...base,
      kind: 'function',
      glslName,
      paramTypes,
      reads: [...c.samplers],
      glsl: c.samplers.size ? undefined : bind({}),
      bind,
      landmarks: [...c.literals],
      parameterReads: [...c.reads],
    });
  }
  return mirrors;
}

/**
 * Every annotated mirror in the shader sources. The sources without `#version` are the pieces the image-formation
 * programs are assembled from (each is GLSL_COMMON plus the heart, the thorax or a pass), so they are read as one
 * program, where a name two pieces define differently (each pass's main) is left untranslated; each `#version` source
 * (Doppler, image chain) is a program of its own, with its own constants. One annotation found in two pieces (a module
 * and the program that includes it) is one mirror.
 */
export function glslMirrors(sources: readonly string[]): GlslMirror[] {
  const isProgram = (s: string) => /^\s*#version/.test(s);
  const programs = [sources.filter((s) => !isProgram(s)), ...sources.filter(isProgram).map((s) => [s])];
  const byKey = new Map<string, GlslMirror>();
  for (const m of programs.flatMap(programMirrors)) {
    const key = `${m.tsPath}: ${m.tsName} ← ${m.glslName}`;
    const old = byKey.get(key);
    if (!old || (old.reason && !m.reason)) byKey.set(key, m);
  }
  return [...byKey.values()];
}

/**
 * Largest difference between a TypeScript original and its GLSL mirror that is not a divergence, relative to
 * max(1, |original|): 2⁻²⁰ ≈ 9.5e-7. The translation runs in double precision, where two copies of one formula can
 * differ only by reassociation and by constants printed with fewer digits. The shader's PI, 3.14159265358979, is the
 * one such constant today (the same float32 as Math.PI): through it four mirrors differ by 1.4e-10 at most, and the
 * other fifty-four agree to the last bit. The GPU evaluates in float32, whose unit roundoff is 2⁻²⁴ ≈ 6e-8: sixteen of them
 * at the result's scale (1 cm or 1 unit at least) are below what the GPU can show, and a coefficient, branch or
 * constant that moves the result by more is a divergence. The GPU's own float32 rounding is not modelled here.
 */
export const MIRROR_TOLERANCE = 2 ** -20;

/**
 * Largest share of a mirror's samples that may be uninformative (NaN on both sides, see `mirrorDeviation`): 1/16. Both
 * copies of a formula give NaN where it leaves its domain (smin's blend radius 0, a zero ellipse ratio), and the GPU's
 * min and max may give a number there instead (IEEE minNum), so such a sample shows nothing either way. Measured when
 * written: at most 169 of 4 096 (4.1 %, sdEllipsoid, where a semi-axis is 0), 69 among the scalar mirrors (smax).
 */
export const MIRROR_UNINFORMATIVE_SHARE = 1 / 16;

/**
 * Difference between the original's and the mirror's value, relative to max(1, |original|), the largest over the
 * components of a vector; NaN when both are NaN (component by component: a sample that tells nothing, which the caller
 * counts against MIRROR_UNINFORMATIVE_SHARE); Infinity when one is NaN, infinite or of another shape.
 */
export function mirrorDeviation(original: GlslValue, mirror: GlslValue): number {
  if (Array.isArray(original) || Array.isArray(mirror)) {
    if (!Array.isArray(original) || !Array.isArray(mirror) || original.length !== mirror.length) return Infinity;
    let worst = NaN;
    original.forEach((o, i) => {
      const d = mirrorDeviation(o, mirror[i]!);
      if (!Number.isNaN(d)) worst = Number.isNaN(worst) ? d : Math.max(worst, d);
    });
    return worst;
  }
  if (original === mirror) return 0;
  if (typeof original !== 'number' || typeof mirror !== 'number') return Infinity;
  if (Number.isNaN(original) && Number.isNaN(mirror)) return NaN;
  if (!Number.isFinite(original) || !Number.isFinite(mirror)) return Infinity;
  return Math.abs(original - mirror) / Math.max(1, Math.abs(original));
}

/**
 * Deterministic arguments for a mirror. Each argument mixes broad ranges, which hold the inputs these functions see in
 * use (fractions and cosines ±1.5, angles and heart coordinates ±8, thorax coordinates ±40 cm), with the function's
 * landmarks (and `extra` ones: a frame's packed values), their negatives, sums and differences, exact and jittered: the
 * values where its branches and clamps turn. An int parameter gets the value rounded, a bool one its sign, a vector one
 * such a value per component.
 */
export function mirrorSamples(m: GlslMirror, count: number, extra: readonly number[] = []): GlslValue[][] {
  let seed = [...m.glslName].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261);
  const random = (): number => {
    // mulberry32
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const marks = new Set<number>();
  const base = [...m.landmarks, ...extra];
  for (const a of base) {
    marks.add(a).add(-a);
    for (const b of base) marks.add(a + b).add(a - b);
  }
  const landmarks = [...marks];
  const value = (): number => {
    const r = random();
    const u = 2 * random() - 1;
    const l = landmarks[Math.floor(random() * landmarks.length)]!;
    const jittered = l + 0.1 * u * Math.max(0.25, Math.abs(l));
    return r < 0.3 ? 1.5 * u : r < 0.45 ? 8 * u : r < 0.6 ? 40 * u : r < 0.75 ? l : jittered;
  };
  const out: GlslValue[][] = [];
  for (let s = 0; s < count; s++) {
    out.push(
      m.paramTypes.map((type) => {
        if (isVec(type)) return Array.from({ length: sizeOf(type) }, () => (isIntType(type) ? Math.round(value()) : value()));
        const v = value();
        return type === 'bool' ? v > 0 : type === 'int' ? Math.round(v) : type === 'uint' ? Math.abs(Math.round(v)) : v;
      }),
    );
  }
  return out;
}
