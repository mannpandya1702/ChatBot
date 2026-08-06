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

use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager, WebviewWindow,
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
        .invoke_handler(tauri::generate_handler![set_click_through, toggle_visibility])
        .setup(|app| {
            let window = app
                .get_webview_window("hud")
                .expect("the hud window is declared in tauri.conf.json");

            // JARVIS_HUD_POSITION is written by the launcher from
            // config.ui.hud_position, so the Rust side needs no config parser.
            let position =
                std::env::var("JARVIS_HUD_POSITION").unwrap_or_else(|_| "bottom-right".into());
            place_window(&window, &position);

            // Click-through until the frontend says otherwise.
            let _ = window.set_ignore_cursor_events(true);

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
