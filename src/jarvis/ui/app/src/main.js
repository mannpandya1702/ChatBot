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

/**
 * Apply the accent colour the user configured.
 *
 * style.css carried a comment saying the accent was "injected from
 * config.ui.accent_color at build time" by an injector that existed nowhere in
 * the tree, so setting it did nothing at all. jarvis.ui.server now writes
 * hud-config.js and this reads it, which means changing the colour is a restart
 * rather than a rebuild.
 */
function applyAccent() {
  const accent = window.JARVIS_ACCENT;
  if (typeof accent === 'string' && /^#[0-9a-fA-F]{3,8}$/.test(accent)) {
    document.documentElement.style.setProperty('--accent', accent);
  }
  return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
}

class Hud {
  constructor() {
    this.orb = new ParticleOrb(document.getElementById('orb'), {
      accentColor: applyAccent(),
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
    this.publishInteractiveRegions();
    this.applyConfiguredPosition();
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

  /**
   * Put the window where the config asks.
   *
   * main.rs reads JARVIS_HUD_POSITION from the environment, with a comment
   * saying "written by the launcher". There is no launcher: nothing in the tree
   * ever set that variable, so every HUD sat bottom-right whatever the config
   * said. The value now arrives in hud-config.js and is handed to the shell,
   * which is the only side that can move a window.
   */
  applyConfiguredPosition() {
    const position = window.JARVIS_HUD_POSITION;
    if (typeof position !== 'string' || !position) return;
    const tauri = window.__TAURI__;
    const invoke = tauri && ((tauri.core && tauri.core.invoke) || tauri.invoke);
    if (!invoke) return;
    invoke('place_hud', { position }).catch(() => {
      /* older shell without the command, or not under Tauri */
    });
  }

  /**
   * Tell the shell where the pointer should be able to reach the HUD.
   *
   * T-3.2 asks for a window that is click-through "except over interactive
   * elements". main.rs set ignore-cursor-events at startup and defined a
   * command to undo it, and nothing ever called that command: not here, not in
   * the built bundle. So the window ignored the cursor permanently and the OS
   * never delivered the mousedown makeDraggable listens for. The drag handle,
   * and everything else, was dead.
   *
   * The obvious repair, a pointermove handler that toggles click-through, does
   * not work and cannot: a window ignoring cursor events receives no pointer
   * events, so the handler would never fire and could never turn itself back
   * on. The hit test has to happen where the cursor is still visible, which is
   * the shell. This side only reports where its controls are.
   *
   * Rectangles are in physical pixels, since that is what the shell compares
   * against the window origin.
   */
  publishInteractiveRegions() {
    const invoke = (() => {
      const tauri = window.__TAURI__;
      if (!tauri) return null;
      // Tauri v2 moved invoke under .core; v1 has it at the top level.
      return (tauri.core && tauri.core.invoke) || tauri.invoke || null;
    })();
    if (!invoke) return;

    const publish = async () => {
      const ratio = window.devicePixelRatio || 1;
      const regions = Array.from(document.querySelectorAll('.interactive'))
        .map((el) => el.getBoundingClientRect())
        .filter((r) => r.width > 0 && r.height > 0)
        .map((r) => ({
          x: r.left * ratio,
          y: r.top * ratio,
          width: r.width * ratio,
          height: r.height * ratio,
        }));
      try {
        await invoke('set_interactive_regions', { regions });
      } catch (err) {
        /* the command is gone or we are not under Tauri after all */
      }
    };

    publish();
    // Republish whenever the layout could have moved: panels appear and hide as
    // state changes, and a stale rectangle is a control that is solid where it
    // no longer is.
    window.addEventListener('resize', publish);
    if (typeof ResizeObserver === 'function') {
      const observer = new ResizeObserver(publish);
      document.querySelectorAll('.interactive').forEach((el) => observer.observe(el));
      this.regionObserver = observer;
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.jarvisHud = new Hud();
});
