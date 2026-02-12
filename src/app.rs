use eframe::egui;
use nalgebra::{Matrix3, Vector3};
use rand::Rng;

use crate::math::snap;
use crate::math::is_near_identity;
use crate::math::matrix_rank_approx;
use crate::math::real_eigenpairs_exact;

use crate::render::draw_grid_3d;
use crate::render::draw_axes_3d;
use crate::render::draw_arrow;
use crate::render::draw_origin_planes;
use crate::render::draw_nav_cube;
use crate::render::draw_determinant_geometry;
use crate::render::draw_eigen_rays;
use crate::render::draw_colored_unit_sphere;
use crate::render::draw_flow_field;

// --- HELPER STRUCT FoR VECTORS ---
#[derive(Clone)]
pub struct UserVector {
    pub pos: Vector3<f32>,
    pub color: egui::Color32,
    pub visible: bool,
    pub name: String,
}

pub struct MatrixApp {
    current: Matrix3<f32>,
    start: Matrix3<f32>,
    target: Matrix3<f32>,
    anim_t: f32,
    animating: bool,
    anim_speed: f32,
    input: [[f32; 3]; 3],
    input_buffer: String,

    user_vectors: Vec<UserVector>,
	math_vec_a: usize,
    math_vec_b: usize,
	active_placement: Option<usize>,
    
    history: Vec<Matrix3<f32>>,

    // Viewport State
    view_rot: f32,      // Yaw
    view_pitch: f32,    // Pitch
    view_zoom: f32,     // Zoom level

    // Draw toggles
    draw_cross_vector: bool,
    draw_parallelogram: bool,
    draw_determinant: bool,
    draw_planes: bool,

    // Eigen visualizations
    draw_flow_field: bool,
    draw_eigen_rays: bool,
    draw_unit_sphere: bool,

    perspective: bool,
    grid_size: i32,
    grid_opacity: u8,
    cross_reverse_order: bool, 
}

impl Default for MatrixApp {
    fn default() -> Self {
        let mut vectors = Vec::new();
        let colors = [
            (egui::Color32::YELLOW, "Yellow (1)"),
            (egui::Color32::from_rgb(160, 32, 240), "Purple (2)"),
            (egui::Color32::from_rgb(0, 255, 255), "Cyan (3)"),
            (egui::Color32::from_rgb(255, 100, 100), "Red (4)"),
            (egui::Color32::from_rgb(100, 255, 100), "Green (5)"),
            (egui::Color32::from_rgb(255, 165, 0), "Orange (6)"),
            (egui::Color32::from_rgb(0, 100, 255), "Blue (7)"),
            (egui::Color32::from_rgb(255, 105, 180), "Pink (8)"),
            (egui::Color32::from_rgb(128, 255, 0), "Lime (9)"),
            (egui::Color32::WHITE, "White (0)"),
        ];

        for (_i, (col, name)) in colors.iter().enumerate() {
            vectors.push(UserVector {
                pos: Vector3::new(0.0, 0.0, 0.0),
                color: *col,
                visible: false,
                name: name.to_string(),
            });
        }

        Self {
            current: Matrix3::identity(),
            start: Matrix3::identity(),
            target: Matrix3::identity(),
            anim_t: 0.0,
            animating: false,
            anim_speed: 1.0,
            input: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            history: Vec::new(),
            view_rot: 0.5,
            view_pitch: 0.3,
            view_zoom: 1.0,
            input_buffer: String::new(),
            
            draw_planes: true,
            draw_parallelogram: false,
            draw_cross_vector: false,
            draw_determinant: true,

            draw_eigen_rays: false,
            draw_flow_field: false,
            draw_unit_sphere: false,

            perspective: true,
            grid_size: 5,
            grid_opacity: 30,
            
            user_vectors: vectors,
			active_placement: None,
			math_vec_a: 0,
			math_vec_b: 1,
            cross_reverse_order: false,
        }
    }
}

impl MatrixApp {
    fn recalculate_target(&mut self) {
        let mut new_target = Matrix3::identity();
        for mat in &self.history {
            new_target = mat * new_target;
        }
        self.start = self.current;
        self.target = new_target;
        self.anim_t = 0.0;
        self.animating = true;
    }

    fn handle_buffered_input(ui: &mut egui::Ui, id: egui::Id, buffer: &mut String, val: &mut f32) {
        let mut display_str = if ui.memory(|mem| mem.has_focus(id)) {
            buffer.clone()
        } else {
            format!("{:.3}", val)
        };

        let response = ui.add(egui::TextEdit::singleline(&mut display_str).id(id).desired_width(50.0));

        if response.gained_focus() {
            *buffer = format!("{:.3}", val);
            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
                let c_range = egui::text::CCursorRange::two(
                    egui::text::CCursor::new(0),
                    egui::text::CCursor::new(buffer.len()),
                );
                state.cursor.set_char_range(Some(c_range));
                egui::TextEdit::store_state(ui.ctx(), id, state);
            }
        }

        if response.changed() {
            *buffer = display_str;
            if let Ok(parsed) = buffer.parse::<f32>() {
                *val = parsed;
            }
        }
    }

    fn swap_columns(&mut self, c1: usize, c2: usize) {
        for r in 0..3 {
            let temp = self.input[r][c1];
            self.input[r][c1] = self.input[r][c2];
            self.input[r][c2] = temp;
        }
        self.recalculate_target();
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        self.input.swap(r1, r2);
        self.recalculate_target();
    }

    fn transpose_input(&mut self) {
        for r in 0..3 {
            for c in (r + 1)..3 {
                let temp = self.input[r][c];
                self.input[r][c] = self.input[c][r];
                self.input[c][r] = temp;
            }
        }
        self.recalculate_target();
    }

    pub fn set_view(&mut self, yaw: f32, pitch: f32) {
        self.view_rot = yaw;
        self.view_pitch = pitch;
    }

    fn handle_hotkeys(&mut self, ctx: &egui::Context) {
        // Ignore hotkeys if typing in an input field
        if ctx.wants_keyboard_input() { return; }
        
        let input = ctx.input(|i| i.clone());

        if input.key_pressed(egui::Key::P) { self.draw_planes = !self.draw_planes; }
        if input.key_pressed(egui::Key::V) { self.perspective = !self.perspective; }
        if input.key_pressed(egui::Key::D) { self.draw_determinant = !self.draw_determinant; }
        if input.key_pressed(egui::Key::E) { self.draw_eigen_rays = !self.draw_eigen_rays; }
        if input.key_pressed(egui::Key::U) { self.draw_unit_sphere = !self.draw_unit_sphere; }
        if input.key_pressed(egui::Key::C) { self.history.clear(); self.recalculate_target(); }
        
        if input.key_pressed(egui::Key::A) && !self.animating {
            let m = Matrix3::new(
                self.input[0][0], self.input[0][1], self.input[0][2],
                self.input[1][0], self.input[1][1], self.input[1][2],
                self.input[2][0], self.input[2][1], self.input[2][2],
            );
            self.history.push(m);
            self.recalculate_target();
        }

        if input.key_pressed(egui::Key::O) && !self.animating {
            self.generate_random_orthogonal_matrix();
        }

        if input.key_pressed(egui::Key::R) && !self.animating {
            if input.modifiers.shift {
                self.generate_random_matrix(1.0, 1.0); 
            } else {
                self.generate_random_matrix(3.0, 0.5);
            }
        }
        
        if input.modifiers.command && input.key_pressed(egui::Key::Z) {
            if !self.history.is_empty() { self.history.pop(); self.recalculate_target(); }
        }
    }

    fn get_view_matrix(&self) -> Matrix3<f32> {
        let (cr, sr) = (self.view_rot.cos(), self.view_rot.sin());
        let (cp, sp) = (self.view_pitch.cos(), self.view_pitch.sin());
        Matrix3::new(
            cr, 0.0, sr,
            sr * sp, cp, -cr * sp,
            -sr * cp, sp, cr * cp,
        )
    }

    fn draw_matrix_input_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Transform Matrix");
        ui.add_space(8.0);

        if ui.button("🎲 Generate Random [R]").clicked()  {
            self.generate_random_matrix(3.0, 0.5);
        }
        ui.add_space(8.0);
        if ui.button("🎲 Generate Random Orthogonal [O]").clicked()  {
            self.generate_random_orthogonal_matrix();
        }
        ui.add_space(8.0);

        egui::Grid::new("matrix_input_grid").spacing([8.0, 8.0]).show(ui, |ui| {
            ui.label(""); 
            if ui.button("⟷").on_hover_text("Byt kolumn 1 & 2").clicked() { self.swap_columns(0, 1); }
            if ui.button("⟷").on_hover_text("Byt kolumn 2 & 3").clicked() { self.swap_columns(1, 2); }
            ui.end_row();

            for r in 0..3 {
                for c in 0..3 {
                    let id = ui.make_persistent_id(format!("mat_input_{}_{}", r, c));
                    Self::handle_buffered_input(ui, id, &mut self.input_buffer, &mut self.input[r][c]);
                }
                if r < 2 {
                    if ui.button("↕").on_hover_text(format!("Byt rad {} & {}", r+1, r+2)).clicked() {
                        self.swap_rows(r, r+1);
                    }
                }
                ui.end_row();
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("⟲ Reset").clicked() {
                self.input = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            }
            if ui.button("⬈ Transpose").clicked() {
                self.transpose_input();
            }
        });
    }

    fn draw_matrix(ui: &mut egui::Ui, m: &Matrix3<f32>) {
        for r in 0..3 {
            ui.horizontal(|ui| {
                for c in 0..3 {
                    let val = m[(r, c)];
                    let color = if val.abs() < 0.001 {
                        egui::Color32::DARK_GRAY
                    } else if val > 0.0 {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::LIGHT_RED
                    };
                    ui.colored_label(color, format!("{:>6.2}", val));
                }
            });
        }
    }

    fn place_vector_at_mouse(&mut self, mouse_pos: egui::Pos2, rect: egui::Rect, view_mat: Matrix3<f32>, vec_idx: usize) {
        if vec_idx >= self.user_vectors.len() { return; }
        
        let center = rect.center();
        let base_scale = (rect.width().min(rect.height()) / 25.0) * self.view_zoom;
        let inv_view = view_mat.transpose();
        let dx = (mouse_pos.x - center.x) / base_scale;
        let dy = (center.y - mouse_pos.y) / base_scale;
        let world_dir = inv_view * Vector3::new(dx, dy, 0.0);
    
        self.user_vectors[vec_idx].pos = Vector3::new(world_dir.x, world_dir.y, 0.0);
        self.user_vectors[vec_idx].visible = true; // Make visible if placing
    }

    fn generate_random_matrix(&mut self, range: f32, step: f32) {
        let mut rng = rand::thread_rng();
        let max = (range / step) as i32;
        for r in 0..3 {
            for c in 0..3 {
                let rand_val = rng.gen_range(-max..=max);
                self.input[r][c] = rand_val as f32 * step;
            }
        }
        self.recalculate_target();
    }

    fn generate_random_orthogonal_matrix(&mut self) {
        let mut rng = rand::thread_rng();
        let u1: f32 = rng.r#gen();
        let u2: f32 = rng.r#gen();
        let u3: f32 = rng.r#gen();

        let sqrt1_minus_u1 = (1.0 - u1).sqrt();
        let sqrt_u1 = u1.sqrt();

        let theta1 = 2.0 * std::f32::consts::PI * u2;
        let theta2 = 2.0 * std::f32::consts::PI * u3;

        let w = sqrt1_minus_u1 * theta1.sin();
        let x = sqrt1_minus_u1 * theta1.cos();
        let y = sqrt_u1 * theta2.sin();
        let z = sqrt_u1 * theta2.cos();

        let mut m = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];

        if rng.gen_bool(0.5) {
            for i in 0..3 { m[i][0] = -m[i][0]; }
        }
        self.input = m;
        self.recalculate_target();
    }

    fn resulting_matrix(&mut self, ui: &mut egui::Ui) {
        ui.heading("Resulting Matrix");
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("M");
                    Self::draw_matrix(ui, &self.target);
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("M");
                        ui.label(egui::RichText::new("-1").small().raised());
                    });
                    if let Some(inv) = self.target.try_inverse() {
                        Self::draw_matrix(ui, &inv);
                    } else {
                        ui.colored_label(egui::Color32::LIGHT_RED, "Not invertible");
                    }
                });
            });
        });
    }

    // --- Generic function for drawing vector ui item ---
    fn draw_vector_ui_item(
        ui: &mut egui::Ui,
        idx: usize,
        user_vector: &mut UserVector,
        transformation_matrix: &nalgebra::Matrix3<f32>,
        input_buffer: &mut String,
    ) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut user_vector.visible, "");
            ui.colored_label(user_vector.color, egui::RichText::new(&user_vector.name).strong());
        });

        if user_vector.visible {
            ui.indent("vec_indent", |ui| {
                // Input field
                ui.horizontal(|ui| {
                    for i in 0..3 {
                        let label = ["X", "Y", "Z"][i];
                        ui.label(format!("{}:", label));
                        let id = ui.make_persistent_id(format!("vec_input_{}_{}", idx, i));
                        Self::handle_buffered_input(ui, id, input_buffer, &mut user_vector.pos[i]);
                    }
                });

                // Transformation
                let transformed = *transformation_matrix * user_vector.pos;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("➞").strong());
                    for i in 0..3 {
                        let val = transformed[i];
                        let val_color = if val.abs() < 0.001 { egui::Color32::DARK_GRAY } 
                                        else if val > 0.0 { egui::Color32::LIGHT_GREEN } 
                                        else { egui::Color32::LIGHT_RED };
                        ui.colored_label(val_color, format!("{:>5.2}", val));
                    }
                });
            });
            ui.add_space(4.0);
        }
    }


    fn draw_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls")
            .width_range(300.0..=350.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.heading("Matrix Visualizer");
                    ui.add_space(4.0);

                    ui.collapsing("⌨ Hotkeys", |ui| {
                        ui.label("1-0: Place Vectors");
                        ui.label("P: Planes | V: Persp | A: Apply");
                        ui.label("Ctrl+Z: Undo | C: Clear");
                    });
                
                    ui.separator();
                    ui.checkbox(&mut self.perspective, "🔭 Perspective [V]");
                
                    ui.separator();
                    ui.heading("Feature Toggles");
                    ui.checkbox(&mut self.draw_planes, "🔳 Show Original [P]");
                    ui.checkbox(&mut self.draw_determinant, "🧊 Determinant [D]");
                    ui.checkbox(&mut self.draw_eigen_rays, "✨ Eigen Rays [E]");
                    ui.checkbox(&mut self.draw_flow_field, "🌊 Flow Field");
                    ui.checkbox(&mut self.draw_unit_sphere, "⚪ Unit Sphere Deformation [U]");

                    ui.add_space(6.0);
                    ui.add(egui::Slider::new(&mut self.grid_opacity, 0..=255).text("Grid Alpha"));
                    ui.add(egui::Slider::new(&mut self.grid_size, 1..=20).text("Grid Size"));
                    ui.add(egui::Slider::new(&mut self.anim_speed, 0.1..=3.0).text("Animation Speed"));
                
                    ui.add_space(10.0);
                    self.draw_matrix_input_ui(ui); 
                
                    ui.separator();
                    self.resulting_matrix(ui);
                
                    ui.add_space(4.0);
                    let det = self.target.determinant();
                    let rank = matrix_rank_approx(&self.target, 1e-6);

                    ui.label(format!("det(M) = {:.4}", det));
                    ui.label(format!("rank(M) = {}", rank));
                    if real_eigenpairs_exact(&self.current, 1e-3).is_empty() {
                        ui.colored_label(egui::Color32::GRAY, "No real eigenvectors");
                    }

                    // --- VECTOR LIST UI ---
                    ui.add_space(10.0);
					ui.separator();
					ui.heading("Vectors (Keys 1-9, 0)");

					for i in 0..self.user_vectors.len() {
					    let vec = &mut self.user_vectors[i];

					    if vec.visible {
					        Self::draw_vector_ui_item(
					            ui,
					            i,
					            vec,
					            &self.current,
					            &mut self.input_buffer,
					        );
					    }
					}

					// Menu for selecting math vectors
					ui.add_space(10.0);
					ui.separator();
					ui.heading("Vector Interaction Math");

					ui.horizontal(|ui| {
					    ui.label("Vector A:");
					    egui::ComboBox::from_id_source("math_a")
					        .selected_text(&self.user_vectors[self.math_vec_a].name)
					        .show_ui(ui, |ui| {
					            for i in 0..10 {
					                ui.selectable_value(&mut self.math_vec_a, i, &self.user_vectors[i].name);
					            }
					        });
					});

					ui.horizontal(|ui| {
					    ui.label("Vector B:");
					    egui::ComboBox::from_id_source("math_b")
					        .selected_text(&self.user_vectors[self.math_vec_b].name)
					        .show_ui(ui, |ui| {
					            for i in 0..10 {
					                ui.selectable_value(&mut self.math_vec_b, i, &self.user_vectors[i].name);
					            }
					        });
					});

					// Calculate using chosen vectors
					let v_a_raw = self.user_vectors[self.math_vec_a].pos;
					let v_b_raw = self.user_vectors[self.math_vec_b].pos;
					let v1 = self.current * v_a_raw;
					let v2 = self.current * v_b_raw;

					let cross = if self.cross_reverse_order { v2.cross(&v1) } else { v1.cross(&v2) };
					let dot = v1.dot(&v2);

					ui.checkbox(&mut self.draw_cross_vector, "Show Cross Product (White)");
					ui.checkbox(&mut self.draw_parallelogram, "Show Parallelogram");

					egui::Frame::group(ui.style()).show(ui, |ui| {
					    ui.label(format!("Dot Product: {:.3}", dot));
					    ui.label(format!("Cross Product Len: {:.3}", cross.norm()));
					});
                });
        });
    }
}

fn smoothstep(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }


impl eframe::App for MatrixApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_hotkeys(ctx);
        let dt = ctx.input(|i| i.unstable_dt).max(1e-6);

        let x_col = egui::Color32::from_hex("#83B366").unwrap();
        let y_col = egui::Color32::from_hex("#FF7154").unwrap();
        let z_col = egui::Color32::from_hex("#8BC9D7").unwrap();

        // --- SIDEBAR ---
        self.draw_sidebar(ctx);

        // --- VIEWPORT ---
        egui::CentralPanel::default().show(ctx, |ui| {
            
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
            
            // Input handling view rotation
            if resp.dragged_by(egui::PointerButton::Primary) {
                self.view_rot += resp.drag_delta().x * 0.01;
                self.view_pitch = (self.view_pitch - resp.drag_delta().y * -0.01).clamp(-1.5, 1.5);
            }
            self.view_zoom = (self.view_zoom * (1.0 + ui.input(|i| i.smooth_scroll_delta.y) * 0.001)).clamp(0.1, 10.0);

            // Setup projection
            let painter = ui.painter_at(rect);
            let view_mat = self.get_view_matrix();
            let base_scale = (rect.width().min(rect.height()) / 25.0) * self.view_zoom;
            let perspective_mode = self.perspective; 

            let project = |v: Vector3<f32>| {
                let v_v = view_mat * v;
                let factor = if perspective_mode { 
                    (base_scale * 20.0) / (20.0 - v_v.z).max(0.1) 
                } else { 
                    base_scale 
                };
                rect.center() + egui::vec2(v_v.x * factor, -v_v.y * factor)
            };

            // --- INPUT LOGIC FOR VECTOR PLACEMENT (1-9, 0) ---
			let wants_keyboard = ctx.wants_keyboard_input();
			let input = ctx.input(|i| i.clone());
					
			let keys = [
				egui::Key::Num1, egui::Key::Num2, egui::Key::Num3, egui::Key::Num4, egui::Key::Num5,
				egui::Key::Num6, egui::Key::Num7, egui::Key::Num8, egui::Key::Num9, egui::Key::Num0,
			];
					
			// --- EVENT-BASED START / VISIBILITY TOGGLE ---
			for event in &input.events {
				if let egui::Event::Key { key, pressed, modifiers, .. } = event {
					for (idx, expected) in keys.iter().enumerate() {
						if key == expected && *pressed && !wants_keyboard {
							if modifiers.alt {
								// Alt pressed → start placement
								self.active_placement = Some(idx);
							} else {
								// No Alt → toggle visibility
								self.user_vectors[idx].visible = !self.user_vectors[idx].visible;
							}
						}
					}
				}
			}
			
			// --- STOP PLACEMENT IF ALT IS RELEASED ---
			if !input.modifiers.alt {
				self.active_placement = None;
			}
			
			// --- GRID SNAP FLAG ---
			// Snapping is default (true). Ctrl disables snapping.
			let snap_to_grid = input.modifiers.alt && !input.modifiers.ctrl;
			
			// --- PLACE VECTOR UNDER MOUSE ---
			if let Some(idx) = self.active_placement {
				if resp.hovered() {
					if let Some(mouse_pos) = input.pointer.hover_pos() {
						self.place_vector_at_mouse(mouse_pos, rect, view_mat, idx);
					
						// Snap if needed
						if snap_to_grid {
							let v = &mut self.user_vectors[idx].pos;
							v.x = snap(v.x, 0.5);
							v.y = snap(v.y, 0.5);
						}
					
						ctx.request_repaint();
					}
				}
			}

            
            // Animation
            if self.animating {
                let base_duration = 0.8;
                self.anim_t += dt * self.anim_speed / base_duration;
                let t = smoothstep(self.anim_t.min(1.0));
                self.current = self.start * (1.0 - t) + self.target * t;
                if self.anim_t >= 1.0 { self.current = self.target; self.animating = false; }
            }

            // --- RENDER SCENE ---
            if self.draw_flow_field {
                draw_flow_field(&painter, &project, &self.current, 0.8, 6);
            }

            if self.draw_planes && !self.current.is_identity(1e-10)  { draw_origin_planes(&painter, &project, self.grid_size); }
            
            let grid_c = egui::Color32::from_rgba_unmultiplied(80, 140, 220, self.grid_opacity);
            draw_grid_3d(&painter, &project, &self.current, grid_c, self.grid_size);
            
            if self.draw_determinant {
                draw_determinant_geometry(&painter, &project, &self.current);
            }

            if self.draw_unit_sphere && matrix_rank_approx(&self.current, 1e-6) >= 2 {
                draw_colored_unit_sphere(&painter, &project, &self.current);
            }

            if self.draw_eigen_rays && !is_near_identity(&self.current, 1e-5) {
                draw_eigen_rays(&painter, &project, &self.current);
            }
            
            draw_axes_3d(&painter, &project);
            
            // Render Basis Vectors (XYZ)
            let m = self.current;
            let basis_vectors = [
                (m*Vector3::x(), x_col),
                (m*Vector3::y(), y_col), 
                (m*Vector3::z(), z_col),
            ];
            
            for (v, color) in basis_vectors {
                draw_arrow(&painter, project(Vector3::zeros()), project(v), color);
            }

            if self.draw_parallelogram {
                let v1 = self.current * self.user_vectors[self.math_vec_a].pos;
                let v2 = self.current * self.user_vectors[self.math_vec_b].pos;
                let poly = vec![
                    project(Vector3::zeros()),
                    project(v1),
                    project(v1 + v2),
                    project(v2),
                ];
                painter.add(egui::Shape::convex_polygon(
                    poly,
                    egui::Color32::from_rgba_unmultiplied(200, 200, 200, 30),
                    egui::Stroke::new(1.0, egui::Color32::LIGHT_GRAY),
                ));
            }
            
            if self.draw_cross_vector {
                let v1 = self.current * self.user_vectors[self.math_vec_a].pos;
                let v2 = self.current * self.user_vectors[self.math_vec_b].pos;
                let cross = if self.cross_reverse_order { v2.cross(&v1) } else { v1.cross(&v2) };
                if cross.norm() > 0.001 {
                    draw_arrow(&painter, project(Vector3::zeros()), project(cross), egui::Color32::WHITE);
                }
            }

            // --- RENDER USER VECTORS ---
            for vec in &self.user_vectors {
                if vec.visible {
                    let transformed = m * vec.pos;
                    draw_arrow(&painter, project(Vector3::zeros()), project(transformed), vec.color);
                }
            }

            // UI Elements (Nav Cube)
            draw_nav_cube(ui, &painter, &view_mat, self);
        });

        ctx.request_repaint();
    }
}
