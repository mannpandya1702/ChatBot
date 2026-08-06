/**
 * Three.js particle orb.
 *
 * T-3.3: roughly 2000 points in a BufferGeometry with PointsMaterial and
 * additive blending, per-particle organic drift, displacement and colour driven
 * by audio level, and a distinct palette per assistant state.
 *
 * The drift is computed on the CPU into a preallocated Float32Array rather than
 * in a shader, because the particle count is small and this keeps the whole
 * thing dependency free apart from three itself. Nothing is allocated per frame.
 */

import * as THREE from 'three';

const PARTICLE_COUNT = 2000;
const BASE_RADIUS = 1.0;

/** Palette per state (T-3.3). Colours are linear RGB triples. */
const PALETTES = {
  idle: { core: [0.05, 0.35, 0.45], glow: [0.1, 0.5, 0.6], pulse: 0.15, spin: 0.05 },
  listening: { core: [0.13, 0.83, 0.93], glow: [0.4, 1.0, 1.0], pulse: 0.5, spin: 0.15 },
  thinking: { core: [0.96, 0.62, 0.04], glow: [1.0, 0.8, 0.3], pulse: 0.9, spin: 0.35 },
  speaking: { core: [0.95, 0.98, 1.0], glow: [1.0, 1.0, 1.0], pulse: 0.7, spin: 0.2 },
  error: { core: [0.86, 0.15, 0.15], glow: [1.0, 0.35, 0.3], pulse: 1.2, spin: 0.02 },
};

/** Linear interpolation between two RGB triples. */
function mixColor(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export class ParticleOrb {
  /**
   * @param {HTMLCanvasElement} canvas Target canvas.
   * @param {{accentColor?: string}} options Styling options.
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.state = 'idle';
    this.targetLevel = 0;
    this.level = 0;
    this.time = 0;
    this.running = false;

    // Palette blending is smoothed so a state change is a transition, not a jump.
    this.palette = { ...PALETTES.idle };
    this.targetPalette = PALETTES.idle;
    this.blend = 1;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    this.camera.position.z = 3.2;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x000000, 0);

    this._buildParticles(options.accentColor);
    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _buildParticles(accentColor) {
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    // Per-particle constants, generated once. Keeping them in typed arrays
    // means the animation loop allocates nothing at all.
    this.home = new Float32Array(PARTICLE_COUNT * 3);
    this.phase = new Float32Array(PARTICLE_COUNT);
    this.speed = new Float32Array(PARTICLE_COUNT);
    this.radius = new Float32Array(PARTICLE_COUNT);

    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      // Fibonacci sphere: even coverage without the clustering at the poles
      // that naive spherical sampling produces.
      const t = i / PARTICLE_COUNT;
      const inclination = Math.acos(1 - 2 * t);
      const azimuth = Math.PI * (1 + Math.sqrt(5)) * i;

      const jitter = 0.9 + Math.random() * 0.2;
      const r = BASE_RADIUS * jitter;
      const x = r * Math.sin(inclination) * Math.cos(azimuth);
      const y = r * Math.sin(inclination) * Math.sin(azimuth);
      const z = r * Math.cos(inclination);

      this.home[i * 3] = x;
      this.home[i * 3 + 1] = y;
      this.home[i * 3 + 2] = z;
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      this.phase[i] = Math.random() * Math.PI * 2;
      this.speed[i] = 0.4 + Math.random() * 0.8;
      this.radius[i] = r;
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    this.material = new THREE.PointsMaterial({
      size: 0.028,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.scene.add(this.points);

    if (accentColor) {
      const accent = new THREE.Color(accentColor);
      PALETTES.idle.glow = [accent.r * 0.6, accent.g * 0.6, accent.b * 0.6];
      PALETTES.listening.glow = [accent.r, accent.g, accent.b];
    }
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const width = (parent && parent.clientWidth) || 320;
    const height = (parent && parent.clientHeight) || 320;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Set the assistant state, starting a palette transition.
   * @param {string} state One of idle, listening, thinking, speaking, error.
   */
  setState(state) {
    const next = PALETTES[state] || PALETTES.idle;
    if (next === this.targetPalette) return;
    // Freeze the current blended palette as the transition's starting point.
    this.palette = {
      core: [...this._currentCore()],
      glow: [...this._currentGlow()],
      pulse: this.palette.pulse,
      spin: this.palette.spin,
    };
    this.targetPalette = next;
    this.blend = 0;
    this.state = state;
  }

  _currentCore() {
    return mixColor(this.palette.core, this.targetPalette.core, this.blend);
  }

  _currentGlow() {
    return mixColor(this.palette.glow, this.targetPalette.glow, this.blend);
  }

  /**
   * Set the audio amplitude driving displacement.
   * @param {number} level 0 to 1.
   */
  setLevel(level) {
    this.targetLevel = Math.max(0, Math.min(1, level || 0));
  }

  /** Advance the simulation and draw one frame. */
  update(deltaSeconds) {
    const dt = Math.min(deltaSeconds, 0.1);
    this.time += dt;

    // Smooth the level so a spiky RMS does not make the orb jitter, and let it
    // fall faster than it rises so speech onsets read as sharp.
    const rate = this.targetLevel > this.level ? 12 : 6;
    this.level += (this.targetLevel - this.level) * Math.min(1, rate * dt);
    this.blend = Math.min(1, this.blend + dt * 2.5);

    const core = this._currentCore();
    const glow = this._currentGlow();
    const pulse = this.palette.pulse + (this.targetPalette.pulse - this.palette.pulse) * this.blend;
    const spin = this.palette.spin + (this.targetPalette.spin - this.palette.spin) * this.blend;

    const positions = this.geometry.attributes.position.array;
    const colors = this.geometry.attributes.color.array;

    const breathe = Math.sin(this.time * 1.1) * 0.02;
    const displacement = 0.12 + this.level * 0.45;
    const thinkingWave = this.state === 'thinking' ? Math.sin(this.time * 4) * 0.05 : 0;

    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      const i3 = i * 3;
      const phase = this.phase[i];
      const speed = this.speed[i];

      // Per-particle sine and cosine drift, which is what makes the cloud look
      // alive rather than like a rotating solid.
      const wobble =
        Math.sin(this.time * speed + phase) * 0.5 + Math.cos(this.time * speed * 0.7 + phase) * 0.5;

      const scale = 1 + breathe + thinkingWave + wobble * displacement * 0.35;
      positions[i3] = this.home[i3] * scale;
      positions[i3 + 1] = this.home[i3 + 1] * scale;
      positions[i3 + 2] = this.home[i3 + 2] * scale;

      // Particles further from the centre pick up the glow colour, which reads
      // as a lit rim.
      const t = Math.min(1, Math.max(0, (wobble + 1) * 0.5 * (0.4 + this.level * 0.6 + pulse * 0.2)));
      colors[i3] = core[0] + (glow[0] - core[0]) * t;
      colors[i3 + 1] = core[1] + (glow[1] - core[1]) * t;
      colors[i3 + 2] = core[2] + (glow[2] - core[2]) * t;
    }

    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;

    this.points.rotation.y += dt * spin;
    this.points.rotation.x = Math.sin(this.time * 0.2) * 0.15;
    this.material.size = 0.024 + this.level * 0.018;

    this.renderer.render(this.scene, this.camera);
  }

  /** Start the animation loop. */
  start() {
    if (this.running) return;
    this.running = true;
    let last = performance.now();
    const frame = (now) => {
      if (!this.running) return;
      const dt = (now - last) / 1000;
      last = now;
      this.update(dt);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  /** Stop the animation loop and release GPU resources. */
  dispose() {
    this.running = false;
    this.geometry.dispose();
    this.material.dispose();
    this.renderer.dispose();
  }
}

export { PARTICLE_COUNT, PALETTES };
