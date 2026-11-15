#!/usr/bin/env python3
""" Error-State EKF for vehicle localization: IMU + wheel encoders + GPS + LiDAR pose. Convention (used consistently everywhere): q_true = q_nominal ⊗ δq(δθ) -> δθ is expressed in the BODY frame R_true = R_nominal @ (I + [δθ]x) Quaternions are Hamilton (w, x, y, z), body -> ENU. """

import numpy as np
from scipy.spatial.transform import Rotation as RSci

I3 = np.eye(3)
I15 = np.eye(15)

# 99% chi-square gates for Mahalanobis outlier rejection
CHI2_99 = {1: 6.63, 2: 9.21, 3: 11.34, 6: 16.81}


# ============================================================
# Helpers
# ============================================================

def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def axis_angle_to_quat(theta):
    angle = np.linalg.norm(theta)
    if angle < 1e-10:
        return np.array([1.0, *(0.5 * theta)])      # first-order, keeps tiny rotations
    s = np.sin(0.5 * angle) / angle
    return np.array([np.cos(0.5 * angle), *(theta * s)])


def quat_to_axis_angle(q):
    """Unit quaternion -> rotation vector (numerically stable near identity)."""
    if q[0] < 0:
        q = -q                                       # shortest path
    n = np.linalg.norm(q[1:])
    if n < 1e-12:
        return 2.0 * q[1:]
    return q[1:] / n * (2.0 * np.arctan2(n, q[0]))


def exp_so3(phi):
    """Rodrigues: rotation vector -> rotation matrix."""
    angle = np.linalg.norm(phi)
    K = skew(phi)
    if angle < 1e-8:
        return I3 + K
    return I3 + (np.sin(angle) / angle) * K + ((1 - np.cos(angle)) / angle**2) * (K @ K)


def quat_to_rotmat(q):
    """Closed form (≈20x faster than building a scipy Rotation each call). q must be unit."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


class GeodeticToENU:
    """WGS84 lat/lon/height -> local ENU around a fixed origin (use the first good fix)."""
    A = 6378137.0
    E2 = 6.69437999014e-3

    def __init__(self, lat0_deg, lon0_deg, h0):
        self.lat0, self.lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
        self.ecef0 = self._ecef(self.lat0, self.lon0, h0)
        sl, cl = np.sin(self.lat0), np.cos(self.lat0)
        so, co = np.sin(self.lon0), np.cos(self.lon0)
        self.R = np.array([[-so, co, 0.0],
                           [-sl * co, -sl * so, cl],
                           [cl * co, cl * so, sl]])

    def _ecef(self, lat, lon, h):
        n = self.A / np.sqrt(1 - self.E2 * np.sin(lat) ** 2)
        return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                         (n + h) * np.cos(lat) * np.sin(lon),
                         (n * (1 - self.E2) + h) * np.sin(lat)])

    def __call__(self, lat_deg, lon_deg, h):
        e = self._ecef(np.radians(lat_deg), np.radians(lon_deg), h)
        return self.R @ (e - self.ecef0)


# ============================================================
# Filter
# ============================================================

class FullErrorStateEKF:
    """ Nominal state : p(3), v(3), q(4), ba(3), bg(3) Error state : δp, δv, δθ(body frame), δba, δbg -> 15-D """

    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.eye(15) * 10.0
        self.g = np.array([0.0, 0.0, -9.81])

        # IMU continuous-time noise densities
        self.sigma_a = 0.004
        self.sigma_w = 0.0001
        self.sigma_ba = 0.001
        self.sigma_bg = 0.00001

        self.gps_lever_arm = np.array([-0.12, 0.0, 0.114])  # Exact Xacro offset (gps_link - imu_link)

        # Measurement noise
        # GPS position noise (1-sigma E, N, U in metres) by fix quality.
        # Typical values; replace with your receiver's real numbers.
        self.R_gps_pos = {
            "single":    np.diag([1.5, 1.5, 3.0]) ** 2,
            "dgps":      np.diag([0.5, 0.5, 1.0]) ** 2,
            "rtk_float": np.diag([0.3, 0.3, 0.6]) ** 2,
            "rtk_fixed": np.diag([0.02, 0.02, 0.04]) ** 2,
        }
        self.R_gps_vel = np.diag([0.1, 0.1, 0.2]) ** 2   # Doppler velocity is much better than position
        self.gps_min_sigma_pos = 0.01                    # floor so a receiver can't claim 0 m error
        self.gps_min_sigma_vel = 0.03
        self.gps_lever_arm = np.zeros(3)                 # antenna position in body frame [m]; SET THIS
        self.gps_max_rejects = 5                         # consecutive gate rejections before forcing an update
        self._gps_rejects = 0
        self._w_last = np.zeros(3)                        # last bias-corrected gyro (for velocity lever arm)
        self.R_wheel = np.diag([0.15, 0.15, 0.3]) ** 2
        self.R_lidar = np.diag([0.08, 0.08, 0.12, 0.02, 0.02, 0.03]) ** 2

        self.wheel_base = 1.6   # unused so far; see notes (wheel yaw-rate update)

        # Preallocated transition matrix: constant blocks are set once
        self._F = np.eye(15)
        self._diag_idx = np.arange(15)

    # ---------------- Prediction ----------------
    def predict(self, accel, gyro, dt):
        if dt <= 0.0:
            return

        a = accel - self.ba
        w = gyro - self.bg
        self._w_last = w
        Rm = quat_to_rotmat(self.q)

        acc_g = Rm @ a + self.g
        self.p = self.p + self.v * dt + 0.5 * acc_g * dt * dt
        self.v = self.v + acc_g * dt
        self.q = quat_normalize(quat_mul(self.q, axis_angle_to_quat(w * dt)))

        # Error-state transition (uses Rm and a from the START of the step)
        F = self._F
        F[0:3, 3:6] = I3 * dt
        F[3:6, 6:9] = -Rm @ skew(a) * dt
        F[3:6, 9:12] = -Rm * dt
        F[6:9, 6:9] = exp_so3(w * dt).T          # exact, instead of I - [w]x dt
        F[6:9, 12:15] = -I3 * dt

        # Q is diagonal -> add to the diagonal instead of building a 15x15 matrix
        q_diag = np.zeros(15)
        q_diag[3:6] = self.sigma_a ** 2 * dt
        q_diag[6:9] = self.sigma_w ** 2 * dt
        q_diag[9:12] = self.sigma_ba ** 2 * dt
        q_diag[12:15] = self.sigma_bg ** 2 * dt

        self.P = F @ self.P @ F.T
        self.P[self._diag_idx, self._diag_idx] += q_diag
        self.P = 0.5 * (self.P + self.P.T)

    # ---------------- Generic update (single code path) ----------------
    def _update(self, y, H, R, gate=None):
        """ Joseph-form EKF update + error injection + ESKF reset. gate: chi-square threshold on the normalised innovation (None = no gating). Returns True if the measurement was accepted. """
        S = H @ self.P @ H.T + R
        if gate is not None and y @ np.linalg.solve(S, y) > gate:
            return False

        K = np.linalg.solve(S, (self.P @ H.T).T).T        # P H^T S^-1 without inv()
        dx = K @ y
        IKH = I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self._inject_error(dx)

        # ESKF reset: re-express covariance in the corrected body frame
        G = np.eye(15)
        G[6:9, 6:9] -= skew(0.5 * dx[6:9])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    # ---------------- GPS ----------------
    @staticmethod
    def _gps_cov(cov, default, min_sigma):
        """Use the receiver's own covariance (3-vector variances or 3x3) if given, else the default."""
        if cov is None:
            R = default.copy()
        else:
            cov = np.asarray(cov, dtype=float)
            R = np.diag(cov) if cov.ndim == 1 else cov.copy()
        d = np.arange(3)
        R[d, d] = np.maximum(R[d, d], min_sigma ** 2)
        return R

    def _gps_update(self, y, H, R, gate):
        """Gated update, but never lock out GPS forever (e.g. after an RTK float->fixed jump)."""
        use_gate = gate if self._gps_rejects < self.gps_max_rejects else None
        ok = self._update(y, H, R, use_gate)
        self._gps_rejects = 0 if ok else self._gps_rejects + 1
        return ok

    def update_gps_position(self, z_pos, fix="single", cov=None, gate=CHI2_99[3]):
        """ z_pos : antenna position in ENU [m] fix : "none" | "single" | "dgps" | "rtk_float" | "rtk_fixed" cov : optional receiver covariance [m²] (3 variances or 3x3); overrides the fix table """
        if fix == "none":
            return False
        R = self._gps_cov(cov, self.R_gps_pos[fix], self.gps_min_sigma_pos)

        Rm = quat_to_rotmat(self.q)
        l = self.gps_lever_arm
        H = np.zeros((3, 15))
        H[:, 0:3] = I3
        H[:, 6:9] = -Rm @ skew(l)                 # antenna offset rotates with attitude error
        y = z_pos - (self.p + Rm @ l)
        return self._gps_update(y, H, R, gate)

    def update_gps_velocity(self, z_vel, fix="single", cov=None, gate=CHI2_99[3]):
        """z_vel : antenna velocity in ENU [m/s] (Doppler)."""
        if fix == "none":
            return False
        R = self._gps_cov(cov, self.R_gps_vel, self.gps_min_sigma_vel)

        Rm = quat_to_rotmat(self.q)
        l = self.gps_lever_arm
        c = np.cross(self._w_last, l)             # antenna velocity due to rotation (body frame)
        H = np.zeros((3, 15))
        H[:, 3:6] = I3
        H[:, 6:9] = -Rm @ skew(c)
        H[:, 12:15] = Rm @ skew(l)                # gyro bias changes the rotational term
        y = z_vel - (self.v + Rm @ c)
        return self._gps_update(y, H, R, gate)

    # ---------------- Wheel encoders ----------------
    def update_wheel_velocity(self, v_left, v_right, gate=CHI2_99[3]):
        """Forward speed from wheels + non-holonomic constraint (no lateral / vertical motion)."""
        v_meas = np.array([0.5 * (v_left + v_right), 0.0, 0.0])

        Rm = quat_to_rotmat(self.q)
        v_pred = Rm.T @ self.v

        H = np.zeros((3, 15))
        H[:, 3:6] = Rm.T
        H[:, 6:9] = skew(v_pred)                 # FIXED: was Rm.T @ skew(v)

        return self._update(v_meas - v_pred, H, self.R_wheel, gate)

    # ---------------- LiDAR pose ----------------
    def update_lidar_pose(self, z_pos, z_rot, gate=CHI2_99[6]):
        z_rot = np.asarray(z_rot, dtype=float)
        if z_rot.shape == (3, 3):
            x, y_, z, w = RSci.from_matrix(z_rot).as_quat()
            q_meas = np.array([w, x, y_, z])
        else:
            q_meas = quat_normalize(z_rot)

        # FIXED: body-frame error q_err = q_est^-1 ⊗ q_meas (was q_meas ⊗ q_est^-1)
        q_err = quat_mul(quat_conj(self.q), q_meas)
        y_rot = quat_to_axis_angle(q_err)

        y = np.concatenate([z_pos - self.p, y_rot])
        H = np.zeros((6, 15))
        H[0:3, 0:3] = I3
        H[3:6, 6:9] = I3
        return self._update(y, H, self.R_lidar, gate)

    # ---------------- Error injection ----------------
    def _inject_error(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        # FIXED: right-multiply (body-frame error), consistent with F and H
        self.q = quat_normalize(quat_mul(self.q, axis_angle_to_quat(dx[6:9])))
        self.ba = self.ba + dx[9:12]
        self.bg = self.bg + dx[12:15]

    # ---------------- Utility ----------------
    def get_state(self):
        return {
            "position": self.p.copy(),
            "velocity": self.v.copy(),
            "quaternion": self.q.copy(),
            "rotmat": quat_to_rotmat(self.q),
            "accel_bias": self.ba.copy(),
            "gyro_bias": self.bg.copy(),
            "covariance": self.P.copy(),
        }

    def set_init_pose(self, p, q=None, v=None):
        self.p = np.asarray(p, dtype=float).copy()      # float cast: int arrays break in-place math
        if v is not None:
            self.v = np.asarray(v, dtype=float).copy()
        if q is not None:
            self.q = quat_normalize(np.asarray(q, dtype=float))
        self.P = np.eye(15)
        self.P[0:3, 0:3] = np.eye(3) * (5.0 ** 2)
        self.P[3:6, 3:6] = np.eye(3) * (2.0 ** 2)
        self.P[6:9, 6:9] = np.eye(3) * ((10 * np.pi / 180) ** 2)
        self.P[9:12, 9:12] = np.eye(3) * (0.2 ** 2)
        self.P[12:15, 12:15] = np.eye(3) * (0.01 ** 2)

    def align_roll_pitch_from_accel(self, accel_static, yaw=0.0):
        """Initialise roll/pitch from a stationary accelerometer average (yaw must come from elsewhere)."""
        ax, ay, az = accel_static
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.hypot(ay, az))
        x, y, z, w = RSci.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
        self.q = np.array([w, x, y, z])