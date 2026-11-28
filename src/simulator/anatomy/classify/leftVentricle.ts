import { Structure, Tissue } from '../tissue';
import { sdRoundCone, smax, smin } from '../sdf';
import { lvCavitySdf } from '../lvShape';
import { fastAtan2, latticeNoise3 } from '@/core/noise';
import { insideMitralOutline, mitralHingeZ, mitralInflowSdf } from '../mitralValve';
import {
  lvTrabecularRelief,
  papillaryFluting,
  septalCrestFactor,
  septalShiftAt,
  wallThicknessAt,
} from '../lvWall';
import { lvSegmentCode, lvWallKind } from '../lvSegments';
import { lvRegionalMotion, regionalCavityOffset } from '../lvGeometry';
import { setSample, type ClassifyCtx } from './context';
import { LV_WALL_BASAL_OVERHANG_CM } from '../heartFrame';

/**
 * LV cavity, papillary muscles, wall and the annular-plane blood. Writes az, levelFrac, the cavity normal, dEllR,
 * wallT and inAnnularRegion for the blocks after it. True when the point is one of these.
 */
export function classifyLeftVentricle(c: ClassifyCtx): boolean {
  const { m, hp, A, x, y, z, out, rootT, rootRr, rootR, inRootLumen, inOutflowLumen } = c;
  const lv = m.lv;
  const zAnn = hp.zAnn;
  const V = hp.valves;
  // local wall thickness by azimuth (septal thicker if IVS > PW) & level, regional motion across the segments
  const az = fastAtan2(y, x);
  const levelFrac = Math.min(1, Math.max(0, (z - zAnn) / Math.max(hp.lengthNow, 1)));
  c.az = az;
  c.levelFrac = levelFrac;
  // septal flattening (D-shape): the septum is pushed toward the LV centre by the RV; the LV ellipsoids are
  // evaluated at x − shift so that both endocardium and epicardium move (the RV crescent uses the same shift)
  const septalShift = septalShiftAt(hp.septalShiftCm, az, levelFrac);
  const xs = x - septalShift;
  const sh = lv.shape;
  const n = c.lvNormal;
  const dProf = lvCavitySdf(hp.prof, sh.ratio, xs, y, z, n);
  const nx0 = n[0]!,
    ny0 = n[1]!,
    nz0 = n[2]!;
  c.nx0 = nx0;
  c.ny0 = ny0;
  c.nz0 = nz0;
  // segment of the tissue here (decision 152)
  const segCode = lvSegmentCode(az, levelFrac, A.rvAzA, A.rvAzP);
  // Regional wall motion, continuous across the segment borders: the share of the normal excursion the wall makes here
  // (1 when the case has no regional abnormality). An akinetic wall keeps its end-diastolic radius → the local cavity SDF
  // is shifted outward by the excursion it withholds, and the wall keeps its end-diastolic thickness (wallThicknessAt).
  const motion = m.regionalMeanFrac > 0 ? lvRegionalMotion(m.segAmp, az, levelFrac) : 1;
  const regional = regionalCavityOffset(lv, hp.rMax, az, levelFrac, motion);
  const rsc = hp.radialScale;
  const lsc = hp.longScale;
  // The wall's own radial scale: its endocardium is at that share of its end-diastolic radius, so its material
  // coordinates (speckle, trabeculae) stay with the tissue. With the global scale an akinetic wall, which stays where it
  // was, read material points up to 45 % further out at end-systole, and its speckle swept through it as if it moved.
  const rscWall = rsc + (1 - motion) * (1 - rsc);
  // trabeculation: rough endocardium with multi-scale longitudinal/anastomosing ridges (material coordinates, so it
  // moves with the wall), growing from the mid cavity to the apex and heavier on non-septal/apical walls
  const mxW = x / rscWall,
    myW = y / rscWall,
    mzW = (z - lv.lengthCm) / lsc;
  const trab =
    levelFrac > 0.45
      ? lvTrabecularRelief(
          az,
          mzW,
          levelFrac,
          latticeNoise3(mxW * 2.6 + 11.3, myW * 2.6 + 2.9, mzW * 1.1 + 6.1, m.wallNoise) - 0.5,
        )
      : 0;
  // Radial regional motion cannot displace the basal clipping plane into the atrium.
  // Apply deformation before intersecting the ventricular half-space (decision 291).
  const dCavR = smax(dProf - regional + trab, zAnn - z, 0.6);
  const tFull = wallThicknessAt(m, hp.thickK, az, levelFrac, motion);
  // the septum narrows under the membranous septum (decision 223); the atrium and the ventricle take the space
  const tNow = tFull * septalCrestFactor(az, z - zAnn);
  c.crestLoss = tFull - tNow;

  // Mitral inflow: the ventricle opens onto the whole annulus. The bullet profile is centred on the long axis and the
  // annulus 0.9 cm behind it, so the posterior and commissural hinges used to lie 0.2-0.8 cm (diastole) and up to
  // 1.2 cm (systole) inside the wall: the posterior leaflet grew out of myocardium and its insertion was lost. The
  // cavity and the wall around it are the smooth union of the profile with the annular outline, which narrows apically
  // into the profile (inflowTaper); basal to the hinges the column is atrium. The outflow tract and root keep their own
  // geometry.
  const inRootTube = rootT > -1.6 && rootRr < rootR + 0.2;
  const zHinge = mitralHingeZ(x, y, V.mitral);
  c.zHinge = zHinge;
  const dInflow = inRootTube ? 1e3 : mitralInflowSdf(x, y, z, V.mitral);
  const dLvBlood = smin(dCavR, dInflow, 0.3);
  // these are read by the aortic root, atria and pericardium blocks even when this one classifies nothing
  const wallT = tNow;
  // the regional offset moves the profile, as in the cavity above, not the inflow column: offset after the union, it
  // pushed the column out too, and at end-systole the akinetic basal inferior wall began 1-5 mm outside the blood, the
  // gap between them filled by the epicardial fat of the blocks below
  const dEllR = smin(dProf - regional, dInflow, 0.3);
  c.wallT = wallT;
  c.dEllR = dEllR;
  c.inAnnularRegion = dEllR < 0 && z < zAnn && !inRootLumen;
  // Papillary muscles inside the cavity (round cones rooted in the wall, see computeHeartPose)
  if (dLvBlood < 0) {
    const P = hp.paps;
    const dPa = sdRoundCone(x, y, z, P[0]!, P[1]!, P[2]!, P[3]!, P[4]!, P[5]!, P[6]!, P[7]!);
    const dPm = sdRoundCone(x, y, z, P[8]!, P[9]!, P[10]!, P[11]!, P[12]!, P[13]!, P[14]!, P[15]!);
    const dPap = Math.min(dPa, dPm) + papillaryFluting(x / rsc, y / rsc, z / lsc);
    if (dPap < 0) {
      setSample(
        out,
        Tissue.Myocardium,
        dPap,
        x,
        y,
        0,
        x / rsc,
        y / rsc,
        z / lsc,
        0,
        Structure.PapillaryMuscle,
      );
      return true;
    }
    // LV blood (the inflow column basal to the hinge plane belongs to the atrium)
    setSample(
      out,
      Tissue.Blood,
      dLvBlood,
      nx0,
      ny0,
      nz0,
      x / rsc,
      y / rsc,
      (z - lv.lengthCm) / lsc,
      0,
      dCavR >= 0 && z < zHinge ? Structure.LaCavity : Structure.LvCavity,
    );
    return true;
  }
  // Ventricular wall: shell of local thickness around the *unclipped* profile, apical to (slightly above)
  // the hinge at this azimuth (decision 327): the annulus is a saddle, and a wall that ended at the mean annular plane
  // hung 3-4 mm past the septal hinge into the left atrium in the four-chamber view, under the hinge, where the
  // commissural hinges sit apical of the plane. The annular plane itself is not a wall: it holds the mitral orifice,
  // the LVOT and fibrous tissue.
  // the trabeculated inner surface belongs to the wall: from the rough endocardium to the smooth epicardium
  if (
    dEllR + trab >= 0 &&
    dEllR < wallT &&
    z >= zHinge - LV_WALL_BASAL_OVERHANG_CM &&
    !inOutflowLumen
  ) {
    // the wall label follows the AHA wall of the segment, bounded by the RV insertions (decision 154)
    const wall = lvWallKind(segCode);
    const structure =
      z > lv.lengthCm - 0.6
        ? Structure.LvApex
        : wall === 1
          ? Structure.LvWallSeptal
          : wall === 2
            ? Structure.LvWallAnterior
            : wall === 3
              ? Structure.LvWallInferior
              : Structure.LvWallLateral;
    const nearEpi = wallT - dEllR < dEllR;
    // The only coherent interface of the wall is its smooth epicardium: the trabeculated endocardium is rough at the
    // wavelength and scatters, it does not reflect. Until decision 144 the sample distance reached the endocardium too
    // and the renderer drew a specular line along it, so the wall read brightest at its inner edge (1.24-1.32 of the
    // mid-wall grey against 0.80-0.84 in CAMUS Good, whose edge pixels mix with blood).
    const dIn = nearEpi ? -(wallT - dEllR) : -wallT;
    const sign = nearEpi ? 1 : -1;
    setSample(
      out,
      Tissue.Myocardium,
      dIn,
      sign * nx0,
      sign * ny0,
      sign * nz0,
      x / rscWall,
      y / rscWall,
      (z - lv.lengthCm) / lsc,
      0,
      structure,
    );
    // where across the wall the sample lies, for the fibre helix of the renderer (decision 144)
    out.transmural = Math.min(1, Math.max(0, dEllR / wallT));
    out.segment = segCode;
    return true;
  }
  // Annular plane region (inside the ellipsoid but basal to the annulus): mitral orifice is blood
  // continuous with the LA; the LVOT is handled by the aortic tube next; the rest is fibrous tissue.
  if (c.inAnnularRegion) {
    // the mitral orifice column basal to the annular plane is atrial blood (the LV ends at the annulus); it follows the
    // D-shaped annulus, so in front of the straight segment the aortomitral curtain and the outflow tract remain
    if (insideMitralOutline(x, y, V.mitral)) {
      setSample(out, Tissue.Blood, -0.3, 0, 0, 1, x, y, z, 0, Structure.LaCavity);
      return true;
    }
  }
  return false;
}
