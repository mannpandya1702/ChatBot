/**
 * HUD entry point.
 *
 * T-3.5: wire the orb and the panels to the WebSocket, with reconnect on drop.
 * The core may restart at any time, so the socket reconnects with backoff and
 * the HUD shows a disconnected state rather than freezing on stale data.
 */

import { ParticleOrb } from './orb.js';
import { Sparkline, TranscriptLog } from './panels.js';

const WS_URL = `ws://${window.JARVIS_HOST || '127.0.0.1'}:${window.JARVIS_PORT || 8765}`;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 10000;

const STATE_LABELS = {
  idle: 'standing by',
  listening: 'listening',
  thinking: 'thinking',
  speaking: 'speaking',
  error: 'error',
};

class Hud {
  constructor() {
    this.orb = new ParticleOrb(document.getElementById('orb'), {
      accentColor: getComputedStyle(document.documentElement)
        .getPropertyValue('--accent')
        .trim(),
    });
    this.orb.start();

    this.stateLabel = document.getElementById('state-label');
    this.transcriptEl = document.getElementById('transcript');
    this.responseEl = document.getElementById('response');
    this.toolEl = document.getElementById('tool');
    this.confirmEl = document.getElementById('confirmation');
    this.connectionEl = document.getElementById('connection');

    this.sparklines = {
      cpu: new Sparkline(document.getElementById('spark-cpu'), { label: 'CPU', unit: '%' }),
      memory: new Sparkline(document.getElementById('spark-mem'), { label: 'RAM', unit: '%' }),
      gpu: new Sparkline(document.getElementById('spark-gpu'), { label: 'GPU', unit: '%' }),
      network: new Sparkline(document.getElementById('spark-net'), {
        label: 'NET',
        unit: ' Mb/s',
        autoScale: true,
      }),
    };
    this.log = new TranscriptLog(document.getElementById('history'));

    this.socket = null;
    this.retryDelay = RECONNECT_MIN_MS;
    this.connect();
    this.makeDraggable();
  }

  connect() {
    this.setConnected(false, 'connecting');
    let socket;
    try {
      socket = new WebSocket(WS_URL);
    } catch (err) {
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      // Only reset the backoff once a connection actually succeeds, otherwise a
      // server that accepts and immediately drops would be hammered.
      this.retryDelay = RECONNECT_MIN_MS;
      this.setConnected(true);
    };

    socket.onmessage = (event) => {
      let frame;
      try {
        frame = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      if (frame && frame.type === 'state') this.apply(frame);
    };

    socket.onclose = () => {
      this.setConnected(false, 'reconnecting');
      this.scheduleReconnect();
    };

    socket.onerror = () => {
      try {
        socket.close();
      } catch (err) {
        /* close() on an already-dead socket is not interesting */
      }
    };
  }

  scheduleReconnect() {
    setTimeout(() => this.connect(), this.retryDelay);
    // Exponential backoff, capped, so a core that stays down does not spin.
    this.retryDelay = Math.min(this.retryDelay * 2, RECONNECT_MAX_MS);
  }

  setConnected(connected, note) {
    if (!this.connectionEl) return;
    this.connectionEl.textContent = connected ? '' : note || 'disconnected';
    this.connectionEl.classList.toggle('visible', !connected);
    if (!connected) {
      this.orb.setLevel(0);
      this.orb.setState('idle');
    }
  }

  apply(frame) {
    this.orb.setState(frame.state);
    this.orb.setLevel(frame.audio_level);

    if (this.stateLabel) {
      this.stateLabel.textContent = STATE_LABELS[frame.state] || frame.state;
      this.stateLabel.dataset.state = frame.state;
    }

    const spoken = frame.partial_transcript || frame.transcript || '';
    if (this.transcriptEl) this.transcriptEl.textContent = spoken;
    if (this.responseEl) this.responseEl.textContent = frame.response || '';

    if (this.toolEl) {
      this.toolEl.textContent = frame.tool ? `calling ${frame.tool}` : '';
      this.toolEl.classList.toggle('visible', Boolean(frame.tool));
    }

    if (this.confirmEl) {
      this.confirmEl.textContent = frame.confirmation || frame.error || '';
      this.confirmEl.classList.toggle('visible', Boolean(frame.confirmation || frame.error));
      this.confirmEl.classList.toggle('error', Boolean(frame.error && !frame.confirmation));
    }

    const metrics = frame.metrics || {};
    this.sparklines.cpu.push(metrics.cpu_percent);
    this.sparklines.memory.push(metrics.memory_percent);
    this.sparklines.gpu.push(metrics.gpu_percent);
    this.sparklines.network.push(metrics.net_down_mbps);

    this.log.render(frame.history || []);
  }

  makeDraggable() {
    // T-3.4: the HUD is draggable. Tauri owns the window move, so the handle
    // just asks it to start a drag; in a plain browser this is a no-op.
    const handle = document.getElementById('drag-handle');
    if (!handle) return;
    handle.addEventListener('mousedown', async (event) => {
      if (event.button !== 0) return;
      const tauri = window.__TAURI__;
      if (tauri && tauri.window && tauri.window.getCurrent) {
        try {
          await tauri.window.getCurrent().startDragging();
        } catch (err) {
          /* not running under Tauri */
        }
      }
    });
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.jarvisHud = new Hud();
});
