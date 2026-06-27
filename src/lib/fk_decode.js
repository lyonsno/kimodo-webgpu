/**
 * fk_decode.js — Forward kinematics decoder for Kimodo motion features.
 *
 * Converts raw 369-dim diffusion output to 30-joint 3D positions.
 * Pure JS — no GPU needed.
 *
 * Pipeline: unnormalize → unpack → cont6d_to_matrix → global_to_local_rots → FK
 */

let fkData = null;

export async function loadFKData(url = '/fk_data.json') {
  fkData = await (await fetch(url)).json();
  console.log(`[fk] Loaded: ${fkData.num_joints} joints, ${fkData.joint_names.length} names`);
  return fkData;
}

/**
 * Convert 6D rotation representation to 3x3 rotation matrix.
 * Gram-Schmidt orthogonalization of first two columns.
 * Input: [a1, a2, a3, b1, b2, b3] (two 3D vectors)
 * Output: 3x3 rotation matrix (row-major flat array of 9)
 */
function cont6dToMatrix(rot6d) {
  // First column: normalize a
  let a = [rot6d[0], rot6d[1], rot6d[2]];
  let aNorm = Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]) + 1e-8;
  a = [a[0]/aNorm, a[1]/aNorm, a[2]/aNorm];

  // Second column: b - (b·a)a, then normalize
  let b = [rot6d[3], rot6d[4], rot6d[5]];
  let dot = b[0]*a[0] + b[1]*a[1] + b[2]*a[2];
  b = [b[0] - dot*a[0], b[1] - dot*a[1], b[2] - dot*a[2]];
  let bNorm = Math.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]) + 1e-8;
  b = [b[0]/bNorm, b[1]/bNorm, b[2]/bNorm];

  // Third column: cross product a × b
  const c = [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];

  // Row-major 3x3: columns are a, b, c
  return [
    a[0], b[0], c[0],
    a[1], b[1], c[1],
    a[2], b[2], c[2],
  ];
}

/**
 * Multiply two 3x3 matrices (row-major flat arrays).
 */
function mat3Mul(A, B) {
  const R = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      R[i*3+j] = A[i*3]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j];
    }
  }
  return R;
}

/**
 * Transpose 3x3 matrix.
 */
function mat3T(M) {
  return [M[0],M[3],M[6], M[1],M[4],M[7], M[2],M[5],M[8]];
}

/**
 * Apply 3x3 rotation to 3D vector.
 */
function mat3Vec(M, v) {
  return [
    M[0]*v[0] + M[1]*v[1] + M[2]*v[2],
    M[3]*v[0] + M[4]*v[1] + M[5]*v[2],
    M[6]*v[0] + M[7]*v[1] + M[8]*v[2],
  ];
}

/**
 * Convert global rotation matrices to local (relative to parent).
 */
function globalToLocalRots(globalRots, parents) {
  const n = globalRots.length;
  const localRots = new Array(n);

  for (let i = 0; i < n; i++) {
    const p = parents[i];
    if (p < 0) {
      localRots[i] = globalRots[i]; // root: local = global
    } else {
      // local = parent_global^T @ global
      localRots[i] = mat3Mul(mat3T(globalRots[p]), globalRots[i]);
    }
  }
  return localRots;
}

/**
 * Forward kinematics: local rotations + root position → global joint positions.
 */
function forwardKinematics(localRots, rootPos, neutralJoints, parents) {
  const n = localRots.length;
  const globalTransforms = new Array(n); // each: { rot: 3x3, pos: [3] }
  const jointPositions = new Array(n);

  for (let i = 0; i < n; i++) {
    const p = parents[i];
    const neutral = neutralJoints[i]; // [3] rest pose position

    if (p < 0) {
      // Root joint
      globalTransforms[i] = { rot: localRots[i], pos: [...rootPos] };
    } else {
      const parentT = globalTransforms[p];
      // Offset from parent in rest pose
      const offset = [
        neutral[0] - neutralJoints[p][0],
        neutral[1] - neutralJoints[p][1],
        neutral[2] - neutralJoints[p][2],
      ];
      // Rotated offset
      const rotOffset = mat3Vec(parentT.rot, offset);
      // Global rotation
      const rot = mat3Mul(parentT.rot, localRots[i]);
      // Global position
      const pos = [
        parentT.pos[0] + rotOffset[0],
        parentT.pos[1] + rotOffset[1],
        parentT.pos[2] + rotOffset[2],
      ];
      globalTransforms[i] = { rot, pos };
    }
    jointPositions[i] = globalTransforms[i].pos;
  }

  return jointPositions;
}

/**
 * Decode raw 369-dim motion features to joint positions.
 *
 * @param {number[][]} features - [N, 369] raw (normalized) diffusion output
 * @returns {{ joints: number[][][], parents: number[] }} [N, 30, 3] joint positions
 */
export function decodeMotion(features) {
  if (!fkData) throw new Error('FK data not loaded. Call loadFKData first.');

  const N = features.length;
  const mean = fkData.stats_mean;
  const std = fkData.stats_std;
  const parents = fkData.parents;
  const neutralJoints = fkData.neutral_joints;
  const numJoints = fkData.num_joints;
  const slices = fkData.slice_dict;

  const allJoints = [];

  for (let f = 0; f < N; f++) {
    // Unnormalize
    const feat = features[f].map((v, i) => v * std[i] + mean[i]);

    // Unpack
    const rootPos = feat.slice(slices.smooth_root_pos[0], slices.smooth_root_pos[1]);
    const heading = feat.slice(slices.global_root_heading[0], slices.global_root_heading[1]);
    const localJointsFlat = feat.slice(slices.local_joints_positions[0], slices.local_joints_positions[1]);
    const rotDataFlat = feat.slice(slices.global_rot_data[0], slices.global_rot_data[1]);

    // Decode rotations: [30, 6] → [30, 3x3]
    const globalRots = [];
    for (let j = 0; j < numJoints; j++) {
      const rot6d = rotDataFlat.slice(j * 6, j * 6 + 6);
      globalRots.push(cont6dToMatrix(rot6d));
    }

    // Global → local rotations
    const localRots = globalToLocalRots(globalRots, parents);

    // FK
    const jointPositions = forwardKinematics(localRots, rootPos, neutralJoints, parents);

    allJoints.push(jointPositions);
  }

  return {
    joints: allJoints,
    parents,
    num_frames: N,
    num_joints: numJoints,
  };
}
