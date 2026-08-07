// Jarvis HUD shell.
//
// T-3.2: frameless, transparent, always on top, click-through except over
// interactive elements, no taskbar entry, system tray with show, hide, and quit.
//
// Click-through is the subtle part. The window is set ignore-cursor-events at
// startup so the desktop underneath stays usable, and the frontend toggles it
// off while the pointer is over an interactive element. Without that toggle the
// drag handle and the scrollable transcript would be unreachable.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use std::time::Duration;

use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, WebviewWindow,
};

/// Position the HUD in a screen corner, honouring the configured preference.
fn place_window(window: &WebviewWindow, position: &str) {
    let Ok(Some(monitor)) = window.primary_monitor() else {
        return;
    };
    let screen = monitor.size();
    let scale = monitor.scale_factor();
    let Ok(size) = window.outer_size() else {
        return;
    };

    let margin = (24.0 * scale) as i32;
    let screen_w = screen.width as i32;
    let screen_h = screen.height as i32;
    let win_w = size.width as i32;
    let win_h = size.height as i32;

    let x = match position {
        p if p.ends_with("left") => margin,
        p if p.ends_with("center") => (screen_w - win_w) / 2,
        _ => screen_w - win_w - margin,
    };
    let y = if position.starts_with("top") {
        margin
    } else {
        screen_h - win_h - margin
    };

    let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
}

/// Let the frontend turn click-through on and off as the pointer moves.
#[tauri::command]
fn set_click_through(window: WebviewWindow, ignore: bool) -> Result<(), String> {
    window
        .set_ignore_cursor_events(ignore)
        .map_err(|e| e.to_string())
}

/// One interactive area of the HUD, in physical pixels relative to the window.
#[derive(Clone, Copy, Debug, serde::Deserialize)]
struct Region {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl Region {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

/// Where the frontend publishes the rectangles it wants the pointer to reach.
type Regions = Mutex<Vec<Region>>;

/// Publish the HUD's interactive rectangles for the hit tester.
///
/// The frontend cannot do this hit test itself. A window with
/// `set_ignore_cursor_events(true)` receives no pointer events at all, so a
/// `pointermove` handler would never fire and could never ask for the window to
/// become solid again: the HUD would be click-through permanently, which is
/// exactly the state it shipped in. The frontend therefore reports *where* its
/// controls are and the polling below decides, using the global cursor position,
/// which the OS gives us whether the window is transparent to it or not.
#[tauri::command]
fn set_interactive_regions(app: AppHandle, regions: Vec<Region>) -> Result<(), String> {
    let state = app.state::<Regions>();
    let mut held = state.lock().map_err(|e| e.to_string())?;
    *held = regions;
    Ok(())
}

/// Poll the cursor and make the window solid only over an interactive region.
fn spawn_hit_tester(app: AppHandle) {
    std::thread::spawn(move || {
        // ~60 Hz. The work per tick is two syscalls and a handful of
        // comparisons, and anything slower is visible as a control that takes a
        // moment to wake up under the pointer.
        let interval = Duration::from_millis(16);
        let mut ignoring: Option<bool> = None;

        loop {
            std::thread::sleep(interval);
            let Some(window) = app.get_webview_window("hud") else {
                return;
            };
            // Skip the work entirely while hidden, and make sure the next show
            // re-applies rather than trusting a stale value.
            if !window.is_visible().unwrap_or(false) {
                ignoring = None;
                continue;
            }

            let (Ok(cursor), Ok(origin), Ok(size)) = (
                window.cursor_position(),
                window.outer_position(),
                window.inner_size(),
            ) else {
                continue;
            };

            let local_x = cursor.x - origin.x as f64;
            let local_y = cursor.y - origin.y as f64;
            let inside = local_x >= 0.0
                && local_y >= 0.0
                && local_x < size.width as f64
                && local_y < size.height as f64;

            let over_control = inside
                && app
                    .state::<Regions>()
                    .lock()
                    .map(|regions| regions.iter().any(|r| r.contains(local_x, local_y)))
                    .unwrap_or(false);

            let ignore = !over_control;
            if ignoring != Some(ignore) {
                if window.set_ignore_cursor_events(ignore).is_ok() {
                    ignoring = Some(ignore);
                }
            }
        }
    });
}

/// Move the HUD to a configured corner.
///
/// The setup path reads JARVIS_HUD_POSITION from the environment and its
/// comment says a launcher writes it. There is no launcher, and nothing in the
/// tree ever set that variable, so every HUD sat bottom-right whatever
/// config.ui.hud_position said. The frontend now reads the value out of the
/// config jarvis.ui.server writes and calls this.
#[tauri::command]
fn place_hud(window: WebviewWindow, position: String) -> Result<(), String> {
    place_window(&window, &position);
    Ok(())
}

/// Toggle HUD visibility from the tray.
#[tauri::command]
fn toggle_visibility(window: WebviewWindow) -> Result<(), String> {
    let visible = window.is_visible().map_err(|e| e.to_string())?;
    if visible {
        window.hide().map_err(|e| e.to_string())
    } else {
        window.show().map_err(|e| e.to_string())
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Regions::default())
        .invoke_handler(tauri::generate_handler![
            place_hud,
            set_click_through,
            set_interactive_regions,
            toggle_visibility
        ])
        .setup(|app| {
            let window = app
                .get_webview_window("hud")
                .expect("the hud window is declared in tauri.conf.json");

            // JARVIS_HUD_POSITION is written by the launcher from
            // config.ui.hud_position, so the Rust side needs no config parser.
            let position =
                std::env::var("JARVIS_HUD_POSITION").unwrap_or_else(|_| "bottom-right".into());
            place_window(&window, &position);

            // Click-through until the pointer is over something interactive.
            let _ = window.set_ignore_cursor_events(true);
            spawn_hit_tester(app.handle().clone());

            let show = MenuItem::with_id(app, "show", "Show", true, None::<&str>)?;
            let hide = MenuItem::with_id(app, "hide", "Hide", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &hide, &quit])?;

            TrayIconBuilder::with_id("jarvis-tray")
                .tooltip("Jarvis")
                .menu(&menu)
                .on_menu_event(|app, event| {
                    let Some(window) = app.get_webview_window("hud") else {
                        return;
                    };
                    match event.id.as_ref() {
                        "show" => {
                            let _ = window.show();
                        }
                        "hide" => {
                            let _ = window.hide();
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    }
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to start the Jarvis HUD");
}
