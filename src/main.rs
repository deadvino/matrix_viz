use eframe::egui;

mod app;
mod math;
mod render;


use app::MatrixApp;


fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "3D Matrix Visualizer - Also watch 3b1b",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Box::new(MatrixApp::default())
        }),
    )
}


// --- Helpers ---

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Load emoji-font (fallback)
    fonts.font_data.insert(
        "emoji".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../assets/fonts/NotoEmoji-Regular.ttf"
        )),
    );

    fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .unwrap()
        .push("emoji".to_owned());

    ctx.set_fonts(fonts);
}