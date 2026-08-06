/**
 * HUD panels: metric sparklines and the rolling transcript.
 *
 * T-3.4. Drawn on 2D canvases rather than with DOM nodes, because a sparkline
 * updating at 30 Hz through the DOM would thrash layout.
 */

const HISTORY_LENGTH = 60;

export class Sparkline {
  /**
   * @param {HTMLCanvasElement} canvas Target canvas.
   * @param {{label: string, unit?: string, autoScale?: boolean, max?: number}} options
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.label = options.label || '';
    this.unit = options.unit || '';
    this.autoScale = Boolean(options.autoScale);
    this.max = options.max || 100;
    this.values = [];
    this.latest = null;

    if (canvas) {
      this.ctx = canvas.getContext('2d');
      this._resize();
      window.addEventListener('resize', () => this._resize());
    }
  }

  _resize() {
    if (!this.canvas) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = this.canvas.clientWidth || 160;
    const height = this.canvas.clientHeight || 32;
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    if (this.ctx) this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.draw();
  }

  /**
   * Append a sample and redraw.
   * @param {number|null|undefined} value Null or undefined records a gap.
   */
  push(value) {
    // A null is meaningful: it means the sensor is unavailable, for example a
    // machine with no NVIDIA GPU. It must not be drawn as zero.
    const numeric = value === null || value === undefined ? null : Number(value);
    this.values.push(Number.isFinite(numeric) ? numeric : null);
    while (this.values.length > HISTORY_LENGTH) this.values.shift();
    this.latest = this.values[this.values.length - 1];
    this.draw();
  }

  _scale() {
    if (!this.autoScale) return this.max;
    const finite = this.values.filter((v) => v !== null);
    if (!finite.length) return 1;
    return Math.max(1, Math.max(...finite) * 1.2);
  }

  draw() {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const width = this.canvas.clientWidth || 160;
    const height = this.canvas.clientHeight || 32;
    const accent = getComputedStyle(document.documentElement)
      .getPropertyValue('--accent')
      .trim() || '#22d3ee';

    ctx.clearRect(0, 0, width, height);

    const top = 12;
    const plotHeight = height - top - 2;
    const scale = this._scale();

    ctx.fillStyle = 'rgba(220, 245, 255, 0.55)';
    ctx.font = '9px ui-monospace, monospace';
    ctx.textBaseline = 'top';
    ctx.fillText(this.label, 0, 0);

    if (this.latest === null || this.latest === undefined) {
      ctx.fillStyle = 'rgba(180, 200, 215, 0.4)';
      ctx.textAlign = 'right';
      ctx.fillText('n/a', width, 0);
      ctx.textAlign = 'left';
    } else {
      ctx.fillStyle = accent;
      ctx.textAlign = 'right';
      const shown = this.latest >= 100 ? this.latest.toFixed(0) : this.latest.toFixed(1);
      ctx.fillText(`${shown}${this.unit}`, width, 0);
      ctx.textAlign = 'left';
    }

    if (this.values.length < 2) return;

    const step = width / (HISTORY_LENGTH - 1);
    ctx.beginPath();
    let started = false;
    this.values.forEach((value, index) => {
      if (value === null) {
        started = false;
        return;
      }
      const x = index * step;
      const y = top + plotHeight - (Math.min(value, scale) / scale) * plotHeight;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Fill under the line, but only when the trace is unbroken, since a
    // partial fill across a gap would imply data that is not there.
    if (!this.values.includes(null)) {
      ctx.lineTo(width, top + plotHeight);
      ctx.lineTo(0, top + plotHeight);
      ctx.closePath();
      const gradient = ctx.createLinearGradient(0, top, 0, top + plotHeight);
      gradient.addColorStop(0, `${accent}44`);
      gradient.addColorStop(1, `${accent}00`);
      ctx.fillStyle = gradient;
      ctx.fill();
    }
  }
}

export class TranscriptLog {
  /** @param {HTMLElement} element Container for the rolling transcript. */
  constructor(element) {
    this.element = element;
    this.rendered = '';
  }

  /**
   * Render the conversation history.
   * @param {string[]} lines Newest last.
   */
  render(lines) {
    if (!this.element) return;
    const joined = lines.join('\n');
    // Rebuilding identical DOM at 30 Hz would defeat the point of the diff.
    if (joined === this.rendered) return;
    this.rendered = joined;

    this.element.textContent = '';
    lines.slice(-8).forEach((line) => {
      const row = document.createElement('div');
      const isUser = line.startsWith('you:');
      row.className = `history-line ${isUser ? 'user' : 'assistant'}`;
      row.textContent = line;
      this.element.appendChild(row);
    });
    this.element.scrollTop = this.element.scrollHeight;
  }
}

export { HISTORY_LENGTH };
