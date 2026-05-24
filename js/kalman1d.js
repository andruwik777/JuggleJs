/**
 * 1D Kalman filter with state [position, velocity, acceleration].
 * Same constant-acceleration model as JuggleNet (KalmanFilter.py):
 * F = [[1, dt, 0.5*dt²], [0, 1, dt], [0, 0, 1]], H = [1, 0, 0] — only position is observed.
 *
 * @module kalman1d
 */

/**
 * @typedef {[number, number, number]} KalmanState - [position, velocity, acceleration]
 */

/**
 * One-dimensional Kalman filter for tracking position with velocity and acceleration.
 */
export class Kalman1D {
  /**
   * @param {number} processVariance - Process noise variance (Q diagonal contribution)
   * @param {number} measurementVariance - Measurement noise variance (R)
   */
  constructor(processVariance, measurementVariance) {
    /** @type {KalmanState} State vector [position, velocity, acceleration] */
    this.x = [0, 0, 0];
    /** @type {number[]} 3×3 covariance matrix stored row-major (length 9) */
    this.P = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    /** @type {number[]} Observation matrix H = [1, 0, 0] */
    this.H = [1, 0, 0];
    /** @type {number} Measurement noise variance R */
    this.R = measurementVariance;
    /** @type {number} Process noise variance q */
    this.q = processVariance;
    /** @type {boolean} True after the first {@link Kalman1D#update} call */
    this.initialised = false;
  }

  /**
   * Build state transition matrix F for a time step dt (internal).
   * @param {number} dt - Time step in seconds
   * @private
   */
  setF(dt) {
    const dt2 = 0.5 * dt * dt;
    this.F = [1, dt, dt2, 0, 1, dt, 0, 0, 1];
  }

  /**
   * Correct state with a position measurement.
   * @param {number} z - Observed position
   */
  update(z) {
    const Hx = this.H[0] * this.x[0] + this.H[1] * this.x[1] + this.H[2] * this.x[2];
    const y = z - Hx;
    const HP = [this.P[0], this.P[1], this.P[2]];
    const S = HP[0] + this.R;
    const K = [this.P[0] / S, this.P[3] / S, this.P[6] / S];
    this.x[0] += K[0] * y;
    this.x[1] += K[1] * y;
    this.x[2] += K[2] * y;
    const oneMinusK0 = 1 - K[0];
    this.P[0] = oneMinusK0 * this.P[0];
    this.P[1] = oneMinusK0 * this.P[1];
    this.P[2] = oneMinusK0 * this.P[2];
    this.P[3] = -K[1] * this.P[0] + this.P[3];
    this.P[4] = -K[1] * this.P[1] + this.P[4];
    this.P[5] = -K[1] * this.P[2] + this.P[5];
    this.P[6] = -K[2] * this.P[0] + this.P[6];
    this.P[7] = -K[2] * this.P[1] + this.P[7];
    this.P[8] = -K[2] * this.P[2] + this.P[8];
    this.initialised = true;
  }

  /**
   * Advance state by dt seconds and return predicted position.
   * @param {number} dt - Time step in seconds; if ≤ 0, returns current position without advancing
   * @returns {number} Predicted position (this.x[0] after predict)
   */
  predict(dt) {
    if (dt <= 0) return this.x[0];
    this.setF(dt);
    const F = this.F;
    this.x = [
      F[0] * this.x[0] + F[1] * this.x[1] + F[2] * this.x[2],
      F[3] * this.x[0] + F[4] * this.x[1] + F[5] * this.x[2],
      F[6] * this.x[0] + F[7] * this.x[1] + F[8] * this.x[2]
    ];
    const P = this.P;
    const FP = [
      F[0] * P[0] + F[1] * P[3] + F[2] * P[6], F[0] * P[1] + F[1] * P[4] + F[2] * P[7], F[0] * P[2] + F[1] * P[5] + F[2] * P[8],
      F[3] * P[0] + F[4] * P[3] + F[5] * P[6], F[3] * P[1] + F[4] * P[4] + F[5] * P[7], F[3] * P[2] + F[4] * P[5] + F[5] * P[8],
      F[6] * P[0] + F[7] * P[3] + F[8] * P[6], F[6] * P[1] + F[7] * P[4] + F[8] * P[7], F[6] * P[2] + F[7] * P[5] + F[8] * P[8]
    ];
    this.P = [
      FP[0] * F[0] + FP[1] * F[1] + FP[2] * F[2] + this.q, FP[0] * F[3] + FP[1] * F[4] + FP[2] * F[5], FP[0] * F[6] + FP[1] * F[7] + FP[2] * F[8],
      FP[3] * F[0] + FP[4] * F[1] + FP[5] * F[2], FP[3] * F[3] + FP[4] * F[4] + FP[5] * F[5] + this.q, FP[3] * F[6] + FP[4] * F[7] + FP[5] * F[8],
      FP[6] * F[0] + FP[7] * F[1] + FP[8] * F[2], FP[6] * F[3] + FP[7] * F[4] + FP[8] * F[5], FP[6] * F[6] + FP[7] * F[7] + FP[8] * F[8] + this.q
    ];
    return this.x[0];
  }
}
