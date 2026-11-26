import test from 'node:test';
import assert from 'node:assert/strict';
import {
  glslMirrors,
  GlslUndefined,
  MIRROR_ANNOTATION,
  mirrorDeviation,
  mirrorSamples,
  type GlslMirror,
  type GlslValue,
} from './shaderMirrors';

/**
 * The GLSL → JavaScript translation of the shader mirrors (shaderMirrors.ts) on synthetic programs: what it evaluates
 * as GLSL ES 3.00 does, and what it names instead of translating. cpuGpuParity.test.ts runs it on the real shaders.
 */

/** The mirrors of one synthetic program, by GLSL name (by export when no definition follows the annotation). */
const mirrorsOf = (...sources: string[]): Record<string, GlslMirror> =>
  Object.fromEntries(glslMirrors(sources).map((m) => [m.glslName || m.tsName, m]));

test('scalar GLSL evaluates as written, each parameter sampled as its type', () => {
  const { scalar } = mirrorsOf(`
const float K = 0.25;
// src/a.ts: scalar
float scalar(float x, int n, bool b) {
  float y = b ? x : -x;
  for (int j = 0; j < 2; j++) y += K * float(n + j);
  return smoothstep(0.0, 1.0, y);
}`);
  assert.equal(scalar!.glsl!(-1, 2, true), 0.15625); // y = −1 + 0.25·2 + 0.25·3 = 0.25 → 0.25²·(3 − 0.5)
  assert.equal(scalar!.glsl!(-1, 2, false), 1);
  assert.deepEqual(scalar!.paramTypes, ['float', 'int', 'bool']);
  assert.ok(mirrorSamples(scalar!, 64).every(([, n, b]) => Number.isInteger(n) && typeof b === 'boolean'));
});

test('int / and % truncate as GLSL integer division does; a zero divisor or a negative % is an error', () => {
  const { half, rest, ratio } = mirrorsOf(`
// src/a.ts: half
float half(int n) { int h = n / 2; return float(h); }
// src/a.ts: rest
int rest(int a, int b) { return a % b; }
// src/a.ts: ratio
float ratio(int n) { return float(n) / 2.0; }`);
  // the translation used to divide as floats: 1.5 for 3, where the GPU gives 1
  assert.equal(half!.glsl!(3), 1);
  assert.equal(half!.glsl!(-3), -1);
  assert.equal(ratio!.glsl!(3), 1.5);
  assert.equal(rest!.glsl!(7, 3), 1);
  assert.throws(() => rest!.glsl!(7, 0), /undefined in GLSL/);
  assert.throws(() => rest!.glsl!(-7, 3), /undefined in GLSL/);
  const { zero } = mirrorsOf(`// src/a.ts: zero\nint zero(int n) { return n / (n - n); }`);
  assert.throws(() => zero!.glsl!(4), /division by zero/);
});

test('uint, round, bitwise operators on floats and implicit conversions are named, not translated', () => {
  const m = mirrorsOf(`
// src/a.ts: wraps
float wraps(float x) { uint a = uint(x); a = a - 1u; return float(a); }
// src/a.ts: literal
float literal(float x) { return x + float(1u); }
// src/a.ts: halves
float halves(float x) { return round(x); }
// src/a.ts: floatBits
float floatBits(float x) { return x & 1.0; }
// src/a.ts: mixes
float mixes(int n) { return float(n) * 2.0 + n; }
// src/a.ts: filtered
float filtered(vec2 uv) { return texture(uImage, uv).r; }
uniform sampler2D uImage;`);
  // −1 for x = 0, where the GPU wraps to 4294967295
  assert.match(m.wraps!.reason!, /uint/);
  assert.match(m.literal!.reason!, /uint/);
  // −2 for −2.5, where GLSL ES 3.00 lets the implementation choose
  assert.match(m.halves!.reason!, /round/);
  assert.match(m.floatBits!.reason!, /bitwise & to a float/);
  assert.match(m.mixes!.reason!, /does not convert/);
  assert.match(m.filtered!.reason!, /filtering/);
  for (const name of ['wraps', 'literal', 'halves', 'floatBits', 'mixes', 'filtered']) assert.equal(m[name]!.glsl, undefined);
});

test('int arithmetic keeps the low 32 bits as GLSL ES 3.00 does; what it leaves undefined raises GlslUndefined', () => {
  const m = mirrorsOf(`
// src/a.ts: shifted
float shifted(int n) { int h = n * 65536 * 65536; return float(h); }
// src/a.ts: lcg
float lcg(int n) { int h = n * 1103515245 + 12345; return float(h / 65536); }
// src/a.ts: sum
int sum(int a, int b) { return a + b - 0; }
// src/a.ts: negated
int negated(int a) { return -a; }
// src/a.ts: counted
int counted(int a) { int b = a; b++; int c = b++; return c + 0 * b; }
// src/a.ts: quotient
int quotient(int a, int b) { return a / b; }
// src/a.ts: truncated
int truncated(float x) { return int(x); }
// src/a.ts: pattern
int pattern() { return 0xFFFFFFFF; }`);
  // 65536² is 2³², whose low 32 bits are 0 (the translation used to give 4294967296)
  assert.equal(m.shifted!.glsl!(1), 0);
  // the hash of 7: 7724606715 wraps to −865327877, + 12345, / 65536 (the translation used to give 117868)
  assert.equal(m.lcg!.glsl!(7), -13203);
  assert.equal(m.sum!.glsl!(2147483647, 1), -2147483648);
  assert.equal(m.negated!.glsl!(-2147483648), -2147483648);
  assert.equal(m.counted!.glsl!(2147483646), 2147483647);
  assert.equal(m.counted!.glsl!(2147483647), -2147483648);
  assert.equal(m.pattern!.glsl!(), -1);
  assert.equal(m.quotient!.glsl!(-7, 2), -3);
  assert.throws(() => m.quotient!.glsl!(-2147483648, -1), GlslUndefined);
  assert.throws(() => m.quotient!.glsl!(7, 0), GlslUndefined);
  assert.equal(m.truncated!.glsl!(-2.7), -2);
  assert.throws(() => m.truncated!.glsl!(1e10), GlslUndefined);
  assert.throws(() => m.truncated!.glsl!(NaN), GlslUndefined);
});

test('the bitwise operators act on the int bits, and integer vectors as the noise lattice reads them', () => {
  const m = mirrorsOf(`
// src/a.ts: wrap
ivec3 wrap(vec3 p) { ivec3 i0 = ivec3(floor(p)) & 127; return (i0 + 1) & 127; }
// src/a.ts: bits
int bits(int a, int b) { return ((a | b) ^ (a & b)) + (~a) + (a >> 1) + (b << 2); }
// src/a.ts: shift
int shift(int a, int n) { return a << n; }
// src/a.ts: component
float component(ivec2 v, int k) { return float(v.yx[k]) + vec2(v).x; }`);
  // floor(−3.5) is −4, whose low 7 bits are 124, as core/noise.ts's Math.floor(x) & 127
  assert.deepEqual(m.wrap!.glsl!([-3.5, 126.2, 127.9]), [125, 127, 0]);
  assert.equal(m.bits!.glsl!(-6, 3), ((-6 | 3) ^ (-6 & 3)) + ~-6 + (-6 >> 1) + (3 << 2));
  assert.equal(m.shift!.glsl!(1, 31), -2147483648);
  assert.throws(() => m.shift!.glsl!(1, 32), GlslUndefined);
  assert.throws(() => m.shift!.glsl!(1, -1), GlslUndefined);
  assert.equal(m.component!.glsl!([3, 5], 0), 8);
  assert.deepEqual(m.wrap!.paramTypes, ['vec3']);
  assert.ok(mirrorSamples(m.component!, 16).every(([v]) => (v as number[]).every(Number.isInteger)));
});

test('structs are copied where GLSL copies them: on declaration, on assignment and into an in parameter', () => {
  const m = mirrorsOf(`
struct H { float d; vec2 v; };
void put(out H s, float d) { s.d = d; s.v = vec2(d); }
float bump(H h) { h.d = 100.0; h.v.x = 100.0; return h.d; }
// src/a.ts: declared
float declared(float x) { H a; put(a, x); H b = a; b.d = 7.0; b.v.y = 7.0; return a.d + a.v.y; }
// src/a.ts: assigned
float assigned(float x) { H a; put(a, x); H b; put(b, 1.0); b = a; b.v.x = 9.0; return a.v.x + b.d; }
// src/a.ts: passed
float passed(float x) { H a; put(a, x); float r = bump(a); return a.d + a.v.x + r; }`);
  // GLSL gives 4 for 2: b is a copy of a (the translation aliased the two and gave 14)
  assert.equal(m.declared!.glsl!(2), 4);
  assert.equal(m.assigned!.glsl!(2), 4);
  assert.equal(m.passed!.glsl!(2), 104);
});

test('a struct assigned to itself keeps its fields: a = a, a = c ? a : b, one struct passed to two inout parameters', () => {
  const m = mirrorsOf(`
struct H { float d; vec2 v; };
void put(out H s, float d) { s.d = d; s.v = vec2(d); }
void copyInto(inout H a, inout H b) { a = b; }
// src/a.ts: chosen
float chosen(float x) { H a; put(a, x); H b; put(b, 1.0); a = x > 0.0 ? a : b; return a.d + a.v.y; }
// src/a.ts: itself
float itself(float x) { H a; put(a, x); a = a; return a.d + a.v.y; }
// src/a.ts: twice
float twice(float x) { H a; put(a, x); copyInto(a, a); return a.d + a.v.y; }`);
  // 6 for 3, as GLSL copies the value before it assigns it (the translation emptied a first and lost its fields)
  assert.equal(m.chosen!.glsl!(3), 6);
  assert.equal(m.chosen!.glsl!(-3), 2);
  assert.equal(m.itself!.glsl!(3), 6);
  assert.equal(m.twice!.glsl!(3), 6);
});

test('a constant array of scalars is read by index; one beyond it is undefined; any other array is named', () => {
  const m = mirrorsOf(`
const float COURSE[6] = float[6](0.5, -1.0, 2.0, 0.25, 3.0, -0.5);
const int STEPS[3] = int[3](1, 2, 3);
const float OPEN[] = float[](1.0, 2.0);
// src/a.ts: course
float course(int i, float t) { return COURSE[3 * i] + t * COURSE[3 * i + 2] + float(STEPS[i]); }
// src/a.ts: beyond
float beyond(float x) { return COURSE[6] * x; }
// src/a.ts: whole
float whole(float x) { return COURSE + x; }
// src/a.ts: local
float local(float x) { float a[2]; a[0] = x; return a[0]; }
// src/a.ts: converted
float converted(float x) { return float[2](1.0, 2)[0] * x; }
// src/a.ts: unsized
float unsized(float x) { return OPEN[0] * x; }`);
  // the pulmonary veins' course is read so (PV_COURSE[3 * i], …): elements 3i and 3i + 2
  assert.equal(m.course!.glsl!(0, 2), 0.5 + 2 * 2 + 1);
  assert.equal(m.course!.glsl!(1, 2), 0.25 + 2 * -0.5 + 2);
  // an index beyond the array is undefined (§5.7): a computed one raises GlslUndefined, a constant one is named
  assert.throws(() => m.course!.glsl!(2, 0), GlslUndefined);
  assert.throws(() => m.course!.glsl!(-1, 0), GlslUndefined);
  assert.match(m.beyond!.reason!, /element 6 of a float\[6\]/);
  assert.ok(m.course!.landmarks.includes(0.25), 'the elements are landmarks of the mirror');
  assert.match(m.whole!.reason!, /\+ to a float\[6\] and a float/);
  assert.match(m.local!.reason!, /arrays/);
  assert.match(m.converted!.reason!, /float array from a int/);
  assert.match(m.unsized!.reason!, /sized otherwise than by an int literal/);
});

test('an index of a swizzle reads and writes the swizzled vector; one beyond the vector is undefined', () => {
  const m = mirrorsOf(`
// src/a.ts: written
float written(vec2 a, int k) { vec2 v = a; v.yx[k] = 1.0; return v.x * 10.0 + v.y; }
// src/a.ts: read
float read(vec3 a, int k) { return a[k]; }
// src/a.ts: constant
float constant(vec2 a) { return a.yx[2]; }`);
  // v.yx[0] is v.y (the translation wrote a temporary and left v.y at 0)
  assert.equal(m.written!.glsl!([0, 0], 0), 1);
  assert.equal(m.written!.glsl!([0, 0], 1), 10);
  assert.throws(() => m.written!.glsl!([0, 0], 2), GlslUndefined);
  assert.equal(m.read!.glsl!([4, 5, 6], 2), 6);
  assert.throws(() => m.read!.glsl!([4, 5, 6], 3), GlslUndefined);
  assert.match(m.constant!.reason!, /component 2 of a vec2/);
});

test('a swizzle of a swizzle reads and writes the vector beneath', () => {
  const m = mirrorsOf(`
// src/a.ts: component
float component(float x) { vec3 v = vec3(0.0); v.zyx.x = x; return v.z; }
// src/a.ts: indexed
float indexed(float x) { vec3 v = vec3(0.0); v.zyx.yx[0] = x; return v.y; }
// src/a.ts: compound
float compound(float x) { vec3 v = vec3(1.0, 2.0, 3.0); v.zyx.xy += vec2(x); return v.z * 100.0 + v.y; }
// src/a.ts: read
vec2 read(vec4 a) { return a.wzyx.yx; }
// src/a.ts: repeated
float repeated(float x) { vec3 v = vec3(0.0); v.xxy.yz = vec2(x); return v.x; }`);
  // v.zyx is an l-value of v (§5.5): v.zyx.x is v.z, v.zyx.yx[0] is v.y (the translation wrote a copy and gave 0)
  assert.equal(m.component!.glsl!(5), 5);
  assert.equal(m.indexed!.glsl!(5), 5);
  // v.z and v.y each gain 5: 807 (the translation gave 302)
  assert.equal(m.compound!.glsl!(5), 807);
  assert.deepEqual(m.read!.glsl!([1, 2, 3, 4]), [3, 4]);
  // a swizzle with a repeated component is no l-value, nor what is selected from it
  assert.match(m.repeated!.reason!, /not a variable/);
});

test('the exponential functions outside their domain are undefined, not the numbers JavaScript gives', () => {
  const m = mirrorsOf(`
// src/a.ts: power
float power(float x, float y) { return pow(x, y); }
// src/a.ts: root
float root(float x) { return sqrt(x) + inversesqrt(max(x, 1.0)); }
// src/a.ts: logs
float logs(float x) { return log(x) + log2(x); }
// src/a.ts: inverse
float inverse(float x) { return inversesqrt(x); }`);
  // JavaScript gives 4, 1 and Infinity; GLSL ES 3.00 (§8.2) leaves them undefined
  assert.throws(() => m.power!.glsl!(-2, 2), GlslUndefined);
  assert.throws(() => m.power!.glsl!(0, 0), GlslUndefined);
  assert.throws(() => m.power!.glsl!(0, -1), GlslUndefined);
  assert.equal(m.power!.glsl!(0, 2), 0);
  assert.equal(m.power!.glsl!(4, 0.5), 2);
  assert.throws(() => m.root!.glsl!(-1), GlslUndefined);
  assert.equal(m.root!.glsl!(4), 2.5);
  assert.throws(() => m.logs!.glsl!(0), GlslUndefined);
  assert.throws(() => m.inverse!.glsl!(0), GlslUndefined);
  // a NaN argument is no domain error: it stays NaN, as on both sides
  assert.ok(Number.isNaN(m.root!.glsl!(NaN) as number));
});

test('out and inout parameters are copied back to their arguments on return; an out one starts undefined', () => {
  const m = mirrorsOf(`
struct H { float d; vec2 v; };
void split(float x, out float a, inout vec2 v) { a = x * 2.0; v.x += x; v = v * 1.0; }
float both(float x, out float y) { y = x + 1.0; return x * 2.0; }
void bumped(out float y) { y = y + 1.0; }
void put(out H s, float d) { s.d = d; s.v = vec2(d); }
void onlyD(out H s, float d) { s.d = d; }
void bumpOut(out H s) { s.d = s.d + 1.0; }
// src/a.ts: caller
float caller(float x) { float a = 1.0; vec2 v = vec2(1.0, 2.0); split(x, a, v); float y; float r = both(x, y); return a + v.x + v.y + 10.0 * (r + y); }
// src/a.ts: fresh
float fresh(float x) { float y = x; bumped(y); return y; }
// src/a.ts: swizzled
float swizzled(float x) { vec3 v = vec3(0.0); split(x, v.z, v.yx); return v.x * 100.0 + v.y * 10.0 + v.z; }
// src/a.ts: hit
bool hit(float x, out float y) { if (x > 0.0) { y = x; return true; } y = -x; return false; }
// src/a.ts: partial
float partial(float x) { H a; put(a, x); onlyD(a, 2.0); return a.v.x; }
// src/a.ts: readFirst
float readFirst(float x) { H a; put(a, x); bumpOut(a); return a.d; }
// src/a.ts: unset
float unset(float x) { H a; return a.v.y + x; }`);
  // a = 6, v = (4, 2), r = 6 and y = 4 for 3
  assert.equal(m.caller!.glsl!(3), 6 + 4 + 2 + 100);
  // an out parameter is not the argument's value on entry (GLSL leaves it undefined: NaN here)
  assert.ok(Number.isNaN(m.fresh!.glsl!(5) as number));
  // nor is an out struct the caller's (the translation kept its fields: 3 and 4 for 3), nor a struct declared without a
  // value anything (the translation threw on its missing vector)
  assert.ok(Number.isNaN(m.partial!.glsl!(3) as number));
  assert.ok(Number.isNaN(m.readFirst!.glsl!(3) as number));
  assert.ok(Number.isNaN(m.unset!.glsl!(3) as number));
  // through a swizzle: v.yx = (v.y + x, v.x) and v.z = 2x
  assert.equal(m.swizzled!.glsl!(3), 0 * 100 + 3 * 10 + 6);
  // called directly, such a function gives its value and then its out parameters
  assert.deepEqual(m.hit!.glsl!(-2, 0), [false, 2]);
  assert.deepEqual(m.hit!.paramTypes, ['float', 'float']);
});

test('NaN on both sides is no agreement: the sample tells nothing', () => {
  assert.ok(Number.isNaN(mirrorDeviation(NaN, NaN)));
  assert.equal(mirrorDeviation(1, NaN), Infinity);
  assert.equal(mirrorDeviation(NaN, 0), Infinity);
  assert.equal(mirrorDeviation(1, 1 + 2 ** -21), 2 ** -21);
  assert.equal(mirrorDeviation(true, false), Infinity);
  // a vector: the informative components only, NaN when there is none
  assert.equal(mirrorDeviation([1, NaN], [1, NaN]), 0);
  assert.ok(Number.isNaN(mirrorDeviation([NaN, NaN], [NaN, NaN])));
  assert.equal(mirrorDeviation([1, 2], [1, NaN]), Infinity);
  assert.equal(mirrorDeviation([1, 2], [1, 2, 3]), Infinity);
  // GLSL's min and max may give the number where JavaScript's give NaN (IEEE minNum): such a sample is uninformative
  const { clamp01 } = mirrorsOf('// src/a.ts: clamp01\nfloat clamp01(float x) { return min(1.0, max(0.0, x)); }');
  assert.ok(Number.isNaN(mirrorDeviation(Math.min(1, Math.max(0, NaN)), clamp01!.glsl!(NaN))));
});

test('vectors: component-wise arithmetic, swizzles, constructors, and values copied where GLSL copies them', () => {
  const m = mirrorsOf(`
// src/a.ts: lengthOf
float lengthOf(vec3 p, vec3 c) { vec3 d = (p - c) * 2.0; return length(d.xy) + d.z; }
// src/a.ts: shuffle
vec4 shuffle(vec2 a) { vec4 v = vec4(a, 1.0, 2.0); v.zw = v.yx; v.x += 10.0; return v.wzyx; }
void bump(vec3 p) { p.x = 100.0; }
// src/a.ts: copies
float copies(vec3 p) { vec3 q = p; q.x = 5.0; bump(p); return p.x + q.x; }
// src/a.ts: geometry
vec3 geometry(vec3 a, vec3 b) { return normalize(cross(a, b)) * dot(a, a) + clamp(a, 0.0, 1.0) - vec3(distance(a, b)); }
// src/a.ts: badSwizzle
float badSwizzle(vec2 a) { return a.z; }
// src/a.ts: badMix
vec3 badMix(vec3 a, vec2 b) { return a + b; }`);
  assert.equal(m.lengthOf!.glsl!([3, 4, 1], [0, 0, 0]), 12); // d = (6, 8, 2): |(6, 8)| + 2
  assert.deepEqual(m.shuffle!.glsl!([1, 2]), [1, 2, 2, 11]); // (1, 2, 1, 2) → (1, 2, 2, 1) → (11, 2, 2, 1) → wzyx
  assert.equal(m.copies!.glsl!([1, 2, 3]), 6); // neither q's assignment nor bump's change p
  const g = m.geometry!.glsl!([1, 0, 0], [0, 2, 0]) as number[];
  const dist = Math.sqrt(5);
  assert.deepEqual(g.map((v) => +v.toFixed(12)), [1 - dist, -dist, 1 - dist].map((v) => +v.toFixed(12)));
  assert.match(m.badSwizzle!.reason!, /\.z of a vec2/);
  assert.match(m.badMix!.reason!, /vec3 and a vec2/);
  assert.deepEqual(m.lengthOf!.paramTypes, ['vec3', 'vec3']);
  assert.ok(mirrorSamples(m.shuffle!, 8).every(([a]) => Array.isArray(a) && a.length === 2));
});

test('a texture read is bound to the texels the CPU uploads, through the defines that name them; any function may be replaced', () => {
  const m = mirrorsOf(`
uniform sampler2D uParams;
uniform highp sampler3D uLattice;
float P(int i) { return texelFetch(uParams, ivec2(i >> 2, 0), 0)[i & 3]; }
#define FRAME_GAIN P(3)
#define OFFSET_BASE 5
float inner(float x) { return x * 2.0; }
// src/a.ts: gain
float gain(float x) { return inner(x) * FRAME_GAIN + P(OFFSET_BASE + 1); }
// src/a.ts: fetched
float fetched(vec3 p) { ivec3 i = ivec3(floor(p)) & 1; return texelFetch(uLattice, i, 0).g * 10.0 + float(i.z); }
// src/a.ts: uniformRead
float uniformRead(float x) { return x * uScale; }
uniform float uScale;`);
  const gain = m.gain!;
  assert.deepEqual(gain.reads, ['uParams']);
  assert.equal(gain.glsl, undefined);
  assert.deepEqual(gain.parameterReads, [3]);
  // P(i)'s own GLSL: texel i >> 2 of row 0, component i & 3, of the four floats a texel the CPU packs
  const data = [0, 0, 0, 2, 0, 0, 7, 0];
  const uParams = (c: GlslValue, lod: GlslValue) => {
    const [x, y] = c as number[];
    assert.ok(y === 0 && lod === 0);
    return data.slice(4 * x!, 4 * x! + 4);
  };
  assert.equal(gain.bind!({ uParams })(1.5), 1.5 * 2 * 2 + 7);
  // P itself replaced by the packed array, and a function by the CPU's own approximation of it
  const P = (i: GlslValue) => data[i as number]!;
  assert.equal(gain.bind!({ P, inner: (x) => (x as number) * 10 })(1.5), 1.5 * 10 * 2 + 7);
  assert.throws(() => gain.bind!({})(1), /uParams is a texture the translation reads: bind its texels/);
  // a sampler3D read at integer coordinates (the noise lattice's way)
  const fetched = m.fetched!;
  assert.deepEqual(fetched.reads, ['uLattice']);
  const uLattice = (c: GlslValue) => {
    const [x, y, z] = c as number[];
    return [0, x! + 2 * y! + 4 * z!, 0, 0];
  };
  assert.equal(fetched.bind!({ uLattice })([3.5, -0.5, 1.2]), (1 + 2 * 1 + 4 * 1) * 10 + 1);
  assert.match(m.uniformRead!.reason!, /uniform uScale/);
  // a binding named like a built-in replaces it for every function the mirror reaches (the CPU's fastAtan2 for atan)
  const { angle } = mirrorsOf(`
float halved(float y, float x) { return atan(y, x) * 0.5; }
// src/a.ts: angle
float angle(vec2 v) { return atan(v.y, v.x) + halved(v.y, v.x); }`);
  assert.equal(angle!.glsl!([1, 1]), 1.5 * Math.atan2(1, 1));
  assert.equal(angle!.bind!({ atan: () => 2 })([1, 1]), 3);
});

test('a run of statements reads the variables before it and leaves those it declares, up to its end line', () => {
  const fn = (annotated: string) => `
struct Hit { float d; };
bool classify(vec3 p, out Hit s) {
  float r = length(p);
  float unread = 3.0;
${annotated}
  s.d = d * 2.0;
  return false;
}`;
  const [run] = glslMirrors([
    fn(`  // src/a.ts: part
  float d = r - 1.0;
  vec3 n = p / r;
  if (d < 0.0) {
    s.d = d;
    return true;
  }
  // end of part`),
  ]);
  assert.equal(run!.kind, 'block');
  assert.equal(run!.glslName, 'classify{part}');
  assert.deepEqual([...run!.inputs].sort(), ['p', 'r', 's']);
  assert.deepEqual(run!.outputs, ['d', 'n']);
  const go = run!.bindBlock!({});
  const s: Record<string, unknown> = {};
  const outside = go({ p: [2, 0, 0], r: 2, s });
  assert.equal(outside.returned, undefined);
  assert.deepEqual(outside.values.n, [1, 0, 0]);
  assert.equal(outside.values.d, 1);
  // the statements after the run are not its own
  assert.equal(s.d, undefined);
  const inside = go({ p: [0.5, 0, 0], r: 0.5, s });
  assert.equal(inside.returned, true);
  assert.equal(s.d, -0.5);
  // a run without its end line, one whose end lies inside a statement, one that shadows what it reads
  const [open] = glslMirrors([fn('  // src/a.ts: part\n  float d = r - 1.0;')]);
  assert.match(open!.reason!, /no "\/\/ end of part" closes/);
  const [split] = glslMirrors([fn('  // src/a.ts: part\n  float d = r - 1.0;\n  if (d < 0.0) {\n    // end of part\n    return true;\n  }')]);
  assert.match(split!.reason!, /end lies inside one of its statements/);
  const [shadow] = glslMirrors([
    `bool g(float r) {\n  {\n    // src/a.ts: part\n    float q = r;\n    float r = q;\n    // end of part\n  }\n  return r > 0.0;\n}`,
  ]);
  assert.match(shadow!.reason!, /declares r, which it also reads/);
});

test('an annotation reaches its definition past comments and a precision qualifier; one followed by none is named', () => {
  const src = `
// src/a.ts: share
// share of the SAM a fibre takes at fan fraction q (the comment the annotation used to have to follow)
/* and a block comment */
highp float share(float q) { return 1.0 - q * q; }
// src/a.ts: orphan
const float K = 1.0;
`;
  const m = mirrorsOf(src);
  assert.equal(m.share!.glsl!(0.5), 0.75);
  assert.match(m.orphan!.reason!, /no function definition follows/);
  assert.equal([...src.matchAll(MIRROR_ANNOTATION)].length, 2);
  // one annotation read in a module and in the program that includes it is one mirror
  assert.equal(glslMirrors([src, `${src}\nfloat other() { return 1.0; }`]).length, 2);
});

test('a block mirror reads the variables of its function by name and returns as its function does', () => {
  const [block] = glslMirrors([
    `
struct Hit { float d; vec3 n; };
void put(out Hit s, float d, vec3 n) { s.d = d; s.n = n; }
bool classify(vec3 p, out Hit s) {
  float r = length(p);
  float unread = 3.0;
  vec3 n = p / r;
  // src/a.ts: shell
  {
    if (r < 1.0) {
      put(s, r - 1.0, n);
      return true;
    }
    s.d = r - 1.0;
    n.x = 7.0;
  }
  return false;
}`,
  ]);
  assert.equal(block!.kind, 'block');
  assert.equal(block!.glslName, 'classify{shell}');
  assert.deepEqual([...block!.inputs].sort(), ['n', 'r', 's']);
  const run = block!.bindBlock!({});
  const inside: Record<string, unknown> = {};
  assert.equal(run({ r: 0.5, n: [1, 0, 0], s: inside }).returned, true);
  assert.deepEqual(inside, { d: -0.5, n: [1, 0, 0] });
  const outside: Record<string, unknown> = {};
  const n = [0, 1, 0];
  const r = run({ r: 2, n, s: outside });
  assert.equal(r.returned, undefined);
  assert.equal(outside.d, 1);
  // a vector input is the block's own copy, which it leaves in `values`
  assert.deepEqual(n, [0, 1, 0]);
  assert.deepEqual(r.values.n, [7, 1, 0]);
  assert.throws(() => run({ r: 2, s: {} }), /reads n, which its runner was not given/);
});
