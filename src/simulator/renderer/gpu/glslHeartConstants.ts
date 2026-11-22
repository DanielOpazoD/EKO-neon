import {
  IAS_LEFT_FACE_CM,
  LA_CORNER_ANTERIOR_MARGIN_CM,
  LA_CORNER_FLAT_CM,
  LA_FLOOR_Y_CM,
  LA_POSTERIOR_FLOOR_DROP_CM,
  LA_POSTERIOR_RATIO,
  LA_SEPTAL_FLAT_CM,
  LA_TOUCH_CM,
  LA_LATERAL_HINGE_CM,
  LA_LATERAL_WALL_AHEAD_CM,
  RA_LA_MIN_CM,
} from '@/simulator/anatomy/laGeometry';
import { LV_WALL_BASAL_OVERHANG_CM } from '@/simulator/anatomy/heartFrame';
import { IVC_HEPATIC_SEGMENT_CM } from '@/simulator/anatomy/anchors';
import {
  FOSSA_REACH,
  LA_CORNER_HEIGHT_CM,
  CS_ABOVE_ANNULUS_CM,
  CS_LATERAL_X_CM,
  CS_CRUX_FROM_IAS_CM,
  CS_OSTIUM_FROM_IAS_CM,
  CS_OSTIUM_Y_CM,
  CS_OSTIUM_ABOVE_ANNULUS_CM,
  CS_RADIUS_CM,
  CS_OSTIUM_RADIUS_CM,
  CS_WALL_CM,
  CS_GROOVE_OFFSET_CM,
  CS_ANNULUS_OFFSET_CM,
  RA_ANTEROMEDIAL_BLEND_CM,
  RA_ANTEROMEDIAL_FROM,
  RA_ANTEROMEDIAL_R_CM,
  RA_ANTEROMEDIAL_TO,
  RA_CREST_REACH_CM,
  RA_MEMBRANOUS_R_CM,
  RA_MEMBRANOUS_TO,
  RA_RIGHT_VEIN_CLEARANCE_CM,
  PV_SVC_CLEARANCE_CM,
  RAA_TIP_FROM_AV,
  RAA_BASE_R_CM,
  RAA_TIP_R_CM,
  RAA_ABOVE_FLOOR_CM,
  RA_SEPTAL_RELEASE_CM,
  LA_ROOT_GAP_CM,
  LA_ROOT_NEAR_CM,
  LA_ROOT_TOP_CM,
  LA_ROOT_RELEASE_CM,
  RA_SEPTAL_RELEASE_LA_CM,
  RA_SEPTAL_RELEASE_RAMP_CM,
  RA_ROOF_DESCENT_SHARE,
  RA_LATERAL_HINGE_CM,
  RA_LATERAL_WALL_HOLD,
  RA_LATERAL_WALL_TOP,
  RA_SLEEVE_MARGIN_CM,
  RA_VESTIBULE_SEPTAL_COS,
  RA_VESTIBULE_SEPTAL_RAMP,
  RA_VESTIBULE_UNION_CM,
} from '@/simulator/anatomy/classify/atria';
import { LV_PROF_BINS } from '@/simulator/anatomy/lvShape';
import {
  PV_INF_Z,
  PV_LEFT_INF_T,
  PV_LEFT_SUP_T,
  PV_COURSE,
  PV_OSTIUM_AHEAD_CM,
  PV_RADIUS,
  PV_WALL_CM,
  PV_RIGHT_T,
  PV_SUP_Z,
} from '@/simulator/anatomy/pulmonaryVeins';
import {
  AV_COAPT_HALF,
  ROOT_ASC_T,
  ROOT_EXCURSION,
  ROOT_SINUS_T,
  ROOT_STJ_T,
} from '@/simulator/anatomy/heartModel';
import {
  AV_PHI0,
  LVOT_TAPER_CM,
  AV_LATERAL_COAPTATION_HEIGHT,
  AV_LATERAL_PROFILE_RADIUS,
  AV_CROWN_EXPONENT,
  AV_OPEN_EDGE_FRACTION,
  AV_OPEN_WALL_GAP,
} from '@/simulator/anatomy/aorticValve';
import {
  AML_ARC_EXTENSION,
  CLOSED_DEPTH,
  CLOSED_REACH,
  MV_BINS,
} from '@/simulator/anatomy/mitralValve';
import {
  SKIRT_ABOVE_CM,
  SKIRT_BELOW_CM,
  SKIRT_FIBRE_CLIP,
  SKIRT_LOBE_PERIOD,
  SKIRT_RADIAL_MARGIN_CM,
  SKIRT_TAPER_START,
  SKIRT_TAPER_WIDTH,
  SKIRT_THICK_BASE,
  SKIRT_THICK_BODY,
  SKIRT_THICK_COMMISSURE,
  SKIRT_THICK_EDGE,
  SKIRT_THICK_FLOOR_CM,
  TV_INFLOW_BULGE_CM,
  TV_BUMP_N,
  TV_BUMP_PHI0,
  TV_BUMP_STEP_RAD,
} from '@/simulator/anatomy/valveSkirt';
import { FAR_FROM_HEART_CM } from '@/simulator/anatomy/classify';
import {
  RV_HINGE_INNER_CM,
  RV_HINGE_REACH_CM,
  RV_OUTFLOW_BLEND_CM,
} from '@/simulator/anatomy/classify/rightVentricle';

const f = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

/**
 * Shared GLSL anatomical constants and `Sample` struct / setter for `classifyHeart`.
 * Every numeric constant is projected directly from the TypeScript anatomy modules so CPU
 * and GPU classifiers stay locked to the same geometry.
 */
export const GLSL_HEART_CONSTANTS = /* glsl */ `
const int LV_PROF_BINS = ${LV_PROF_BINS};
const float LA_POSTERIOR_RATIO = ${f(LA_POSTERIOR_RATIO)};
const float LA_POSTERIOR_FLOOR_DROP_CM = ${f(LA_POSTERIOR_FLOOR_DROP_CM)};
const float LA_FLOOR_Y_CM = ${f(LA_FLOOR_Y_CM)};
const float LV_WALL_BASAL_OVERHANG_CM = ${f(LV_WALL_BASAL_OVERHANG_CM)};
const float IVC_HEPATIC_X = ${f(IVC_HEPATIC_SEGMENT_CM[0])};
const float IVC_HEPATIC_Y = ${f(IVC_HEPATIC_SEGMENT_CM[1])};
const float IVC_HEPATIC_Z = ${f(IVC_HEPATIC_SEGMENT_CM[2])};
const float LA_CORNER_HEIGHT_CM = ${f(LA_CORNER_HEIGHT_CM)};
const float FOSSA_REACH = ${f(FOSSA_REACH)};
const float IAS_LEFT_FACE_CM = ${f(IAS_LEFT_FACE_CM)};
const float LA_SEPTAL_FLAT_CM = ${f(LA_SEPTAL_FLAT_CM)};
const float LA_TOUCH_CM = ${f(LA_TOUCH_CM)};
const float LA_LATERAL_HINGE_CM = ${f(LA_LATERAL_HINGE_CM)};
const float LA_LATERAL_WALL_AHEAD_CM = ${f(LA_LATERAL_WALL_AHEAD_CM)};
const float LA_CORNER_FLAT_CM = ${f(LA_CORNER_FLAT_CM)};
const float LA_CORNER_ANTERIOR_MARGIN_CM = ${f(LA_CORNER_ANTERIOR_MARGIN_CM)};
const float RA_LA_MIN_CM = ${f(RA_LA_MIN_CM)};
const float ROOT_SINUS_T = ${f(ROOT_SINUS_T)};
const float LVOT_TAPER_CM = ${f(LVOT_TAPER_CM)};
const float ROOT_STJ_T = ${f(ROOT_STJ_T)};
const float ROOT_ASC_T = ${f(ROOT_ASC_T)};
const float AV_COAPT_HALF = ${f(AV_COAPT_HALF)};
const float ROOT_EXCURSION = ${f(ROOT_EXCURSION)};
const float AV_PHI0 = ${f(AV_PHI0)};
const float AV_LATERAL_COAPTATION_HEIGHT = ${f(AV_LATERAL_COAPTATION_HEIGHT)};
const float AV_LATERAL_PROFILE_RADIUS = ${f(AV_LATERAL_PROFILE_RADIUS)};
const float AV_CROWN_EXPONENT = ${f(AV_CROWN_EXPONENT)};
const float AV_OPEN_EDGE_FRACTION = ${f(AV_OPEN_EDGE_FRACTION)};
const float AV_OPEN_WALL_GAP = ${f(AV_OPEN_WALL_GAP)};
const float RA_ROOF_DESCENT_SHARE = ${f(RA_ROOF_DESCENT_SHARE)};
const float RV_HINGE_INNER_CM = ${f(RV_HINGE_INNER_CM)};
const float RV_HINGE_REACH_CM = ${f(RV_HINGE_REACH_CM)};
const float RA_LATERAL_HINGE_CM = ${f(RA_LATERAL_HINGE_CM)};
const float RA_LATERAL_WALL_HOLD = ${f(RA_LATERAL_WALL_HOLD)};
const float RA_LATERAL_WALL_TOP = ${f(RA_LATERAL_WALL_TOP)};
const float RA_SLEEVE_MARGIN_CM = ${f(RA_SLEEVE_MARGIN_CM)};
const float RA_VESTIBULE_UNION_CM = ${f(RA_VESTIBULE_UNION_CM)};
const float RA_VESTIBULE_SEPTAL_COS = ${f(RA_VESTIBULE_SEPTAL_COS)};
const float RA_VESTIBULE_SEPTAL_RAMP = ${f(RA_VESTIBULE_SEPTAL_RAMP)};
const float RA_ANTEROMEDIAL_FROM_X = ${f(RA_ANTEROMEDIAL_FROM[0])};
const float RA_ANTEROMEDIAL_FROM_Y = ${f(RA_ANTEROMEDIAL_FROM[1])};
const float RA_ANTEROMEDIAL_FROM_Z = ${f(RA_ANTEROMEDIAL_FROM[2])};
const float RA_ANTEROMEDIAL_TO_X = ${f(RA_ANTEROMEDIAL_TO[0])};
const float RA_ANTEROMEDIAL_TO_Y = ${f(RA_ANTEROMEDIAL_TO[1])};
const float RA_ANTEROMEDIAL_TO_Z = ${f(RA_ANTEROMEDIAL_TO[2])};
const float RA_ANTEROMEDIAL_R_CM = ${f(RA_ANTEROMEDIAL_R_CM)};
const float RA_ANTEROMEDIAL_BLEND_CM = ${f(RA_ANTEROMEDIAL_BLEND_CM)};
const float RA_MEMBRANOUS_TO_X = ${f(RA_MEMBRANOUS_TO[0])};
const float RA_MEMBRANOUS_TO_Y = ${f(RA_MEMBRANOUS_TO[1])};
const float RA_MEMBRANOUS_TO_Z = ${f(RA_MEMBRANOUS_TO[2])};
const float RA_MEMBRANOUS_R_CM = ${f(RA_MEMBRANOUS_R_CM)};
const float RA_RIGHT_VEIN_CLEARANCE_CM = ${f(RA_RIGHT_VEIN_CLEARANCE_CM)};
const float PV_SVC_CLEARANCE_CM = ${f(PV_SVC_CLEARANCE_CM)};
const float RAA_TIP_FROM_AV_X = ${f(RAA_TIP_FROM_AV[0])};
const float RAA_TIP_FROM_AV_Y = ${f(RAA_TIP_FROM_AV[1])};
const float RAA_TIP_FROM_AV_Z = ${f(RAA_TIP_FROM_AV[2])};
const float RAA_BASE_R_CM = ${f(RAA_BASE_R_CM)};
const float RAA_TIP_R_CM = ${f(RAA_TIP_R_CM)};
const float RAA_ABOVE_FLOOR_CM = ${f(RAA_ABOVE_FLOOR_CM)};
const float CS_ABOVE_ANNULUS_CM = ${f(CS_ABOVE_ANNULUS_CM)};
const float CS_LATERAL_X_CM = ${f(CS_LATERAL_X_CM)};
const float CS_CRUX_FROM_IAS_CM = ${f(CS_CRUX_FROM_IAS_CM)};
const float CS_OSTIUM_FROM_IAS_CM = ${f(CS_OSTIUM_FROM_IAS_CM)};
const float CS_OSTIUM_Y_CM = ${f(CS_OSTIUM_Y_CM)};
const float CS_OSTIUM_ABOVE_ANNULUS_CM = ${f(CS_OSTIUM_ABOVE_ANNULUS_CM)};
const float CS_RADIUS_CM = ${f(CS_RADIUS_CM)};
const float CS_OSTIUM_RADIUS_CM = ${f(CS_OSTIUM_RADIUS_CM)};
const float CS_WALL_CM = ${f(CS_WALL_CM)};
const float CS_GROOVE_OFFSET_CM = ${f(CS_GROOVE_OFFSET_CM)};
const float PV_OSTIUM_AHEAD_CM = ${f(PV_OSTIUM_AHEAD_CM)};
const float CS_ANNULUS_OFFSET_CM = ${f(CS_ANNULUS_OFFSET_CM)};
const float RA_CREST_REACH_CM = ${f(RA_CREST_REACH_CM)};
const float RA_SEPTAL_RELEASE_LA_CM = ${f(RA_SEPTAL_RELEASE_LA_CM)};
const float RA_SEPTAL_RELEASE_RAMP_CM = ${f(RA_SEPTAL_RELEASE_RAMP_CM)};
const float RA_SEPTAL_RELEASE_CM = ${f(RA_SEPTAL_RELEASE_CM)};
const float LA_ROOT_GAP_CM = ${f(LA_ROOT_GAP_CM)};
const float LA_ROOT_NEAR_CM = ${f(LA_ROOT_NEAR_CM)};
const float LA_ROOT_TOP_CM = ${f(LA_ROOT_TOP_CM)};
const float LA_ROOT_RELEASE_CM = ${f(LA_ROOT_RELEASE_CM)};
const int MV_BINS = ${MV_BINS};
const float AML_ARC_EXTENSION = ${f(AML_ARC_EXTENSION)};
const float MV_CLOSED_REACH[3] = float[3](${CLOSED_REACH.map(f).join(', ')});
const float MV_CLOSED_DEPTH[3] = float[3](${CLOSED_DEPTH.map(f).join(', ')});
const float PV_LEFT_SUP_T = ${f(PV_LEFT_SUP_T)};
const float PV_LEFT_INF_T = ${f(PV_LEFT_INF_T)};
const float PV_RIGHT_T = ${f(PV_RIGHT_T)};
const float PV_SUP_Z = ${f(PV_SUP_Z)};
const float PV_INF_Z = ${f(PV_INF_Z)};
const float PV_COURSE[12] = float[12](${PV_COURSE.map(f).join(', ')});
const float PV_RADIUS = ${f(PV_RADIUS)};
const float PV_WALL_CM = ${f(PV_WALL_CM)};
const float RV_OUTFLOW_BLEND_CM = ${f(RV_OUTFLOW_BLEND_CM)};
const float SKIRT_ABOVE_CM = ${f(SKIRT_ABOVE_CM)};
const float TV_INFLOW_BULGE_CM = ${f(TV_INFLOW_BULGE_CM)};
const float TV_BUMP_STEP_RAD = ${f(TV_BUMP_STEP_RAD)};
const float TV_BUMP_PHI0 = ${f(TV_BUMP_PHI0)};
const int TV_BUMP_N = ${TV_BUMP_N};
const float FAR_FROM_HEART_CM = ${f(FAR_FROM_HEART_CM)};
const float SKIRT_BELOW_CM = ${f(SKIRT_BELOW_CM)};
const float SKIRT_RADIAL_MARGIN_CM = ${f(SKIRT_RADIAL_MARGIN_CM)};
const float SKIRT_FIBRE_CLIP = ${f(SKIRT_FIBRE_CLIP)};
const float SKIRT_TAPER_START = ${f(SKIRT_TAPER_START)};
const float SKIRT_TAPER_WIDTH = ${f(SKIRT_TAPER_WIDTH)};
const float SKIRT_LOBE_PERIOD = ${f(SKIRT_LOBE_PERIOD)};
const float SKIRT_THICK_BASE = ${f(SKIRT_THICK_BASE)};
const float SKIRT_THICK_EDGE = ${f(SKIRT_THICK_EDGE)};
const float SKIRT_THICK_FLOOR_CM = ${f(SKIRT_THICK_FLOOR_CM)};
const float SKIRT_THICK_COMMISSURE = ${f(SKIRT_THICK_COMMISSURE)};
const float SKIRT_THICK_BODY = ${f(SKIRT_THICK_BODY)};
struct Sample {
  int tissue;
  int structure;
  float sdf;
  vec3 n;
  vec3 m;
  float extra;
  float transmural; // depth across the LV wall, 0 endocardium → 1 epicardium; −1 elsewhere (decision 144)
  int segment; // LV segment code of LV compact myocardium (lvSegments.ts, decision 152); 0 elsewhere
};

void setSample(out Sample s, int tissue, float sdf, vec3 n, vec3 m, float extra, int structure) {
  float l = length(n);
  s.tissue = tissue;
  s.sdf = sdf;
  s.n = l > 0.0 ? n / l : n;
  s.m = m;
  s.extra = extra;
  s.structure = structure;
  s.transmural = -1.0;
  s.segment = 0;
}
`;
