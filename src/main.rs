use eframe::{egui, glow::LAYER_PROVOKING_VERTEX};
use nalgebra::{Matrix3, Vector3, zero};
use rand::Rng; // This must be present to use .gen_range()


mod math;

use crate::math::sample_unit_sphere;
use crate::math::is_near_identity;
use crate::math::real_eigenpairs_approx;
use crate::math::best_parallelogram_basis;
use crate::math::matrix_rank_approx;



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

struct MatrixApp {
    current: Matrix3<f32>,
    start: Matrix3<f32>,
    target: Matrix3<f32>,
    anim_t: f32,
    animating: bool,
	anim_speed: f32,
    input: [[f32; 3]; 3],
    input_buffer: String,
    selected_vector: Vector3<f32>,
    history: Vec<Matrix3<f32>>,

    // Viewport State
    view_rot: f32,      // Yaw
    view_pitch: f32,    // Pitch
    view_zoom: f32,     // Zoom level
    click_to_place: bool,

	// Draw toggles
	draw_yellow: bool,
	draw_purple: bool,
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
	grid_opacity: u8, // 0 is invisible, 255 is fully opaque
	selected_vector_purple: Vector3<f32>, // Den nya lila vektorn
    cross_reverse_order: bool,           // För att byta ordning (Lila x Gul vs Gul x Lila)
}

impl Default for MatrixApp {
    fn default() -> Self {
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
            click_to_place: true,
			//draw defaults
            draw_planes: true,
			draw_yellow: false,
			draw_purple: false,
			draw_parallelogram: false,
			draw_cross_vector: false,
			draw_determinant: true,

			draw_eigen_rays: false,
			draw_flow_field: false,
			draw_unit_sphere: false,

            perspective: true,
            grid_size: 5,
			grid_opacity: 30, // A nice subtle default
			selected_vector: Vector3::new(1.0, 1.0, 0.0), // Gul
            selected_vector_purple: Vector3::new(-1.0, 1.0, 0.0), // Lila standard
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

        let response = ui.add(egui::TextEdit::singleline(&mut display_str).id(id).desired_width(60.0));

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


	fn set_view(&mut self, yaw: f32, pitch: f32) {
        self.view_rot = yaw;
        self.view_pitch = pitch;
    }


	fn handle_hotkeys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() { return; }
        let input = ctx.input(|i| i.clone());

        if input.key_pressed(egui::Key::P) { self.draw_planes = !self.draw_planes; }
        if input.key_pressed(egui::Key::V) { self.perspective = !self.perspective; }
		if input.key_pressed(egui::Key::D) { self.draw_determinant = !self.draw_determinant; }
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

		if input.key_pressed(egui::Key::R) && !self.animating {
			if input.modifiers.shift {
		        self.generate_random_matrix(1.0, 1.0); // t.ex. -1..1
		    } else {
		        self.generate_random_matrix(3.0, 0.5);
		    }
		}

		if input.key_pressed(egui::Key::Space) {
			self.click_to_place = true;
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

		// --- NEW RANDOM BUTTON ---
        if ui.button("🎲 Generate Random [R]").clicked()  {
            self.generate_random_matrix(3.0, 0.5);
        }
        ui.add_space(8.0);

        egui::Grid::new("matrix_input_grid").spacing([8.0, 8.0]).show(ui, |ui| {
            // Kolumn-bytar-knappar (Överst)
            ui.label(""); 
            if ui.button("⟷").on_hover_text("Byt kolumn 1 & 2").clicked() { self.swap_columns(0, 1); }
            if ui.button("⟷").on_hover_text("Byt kolumn 2 & 3").clicked() { self.swap_columns(1, 2); }
            ui.end_row();

            for r in 0..3 {
                for c in 0..3 {
                    let id = ui.make_persistent_id(format!("mat_input_{}_{}", r, c));
                    Self::handle_buffered_input(ui, id, &mut self.input_buffer, &mut self.input[r][c]);
                }
                // Rad-bytar-knappar (Till höger)
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


	fn place_vector_at_mouse(&mut self, mouse_pos: egui::Pos2, rect: egui::Rect, view_mat: Matrix3<f32>, target_is_purple: bool) {
	    let center = rect.center();
	    let base_scale = (rect.width().min(rect.height()) / 25.0) * self.view_zoom;
	    let inv_view = view_mat.transpose();
	    let dx = (mouse_pos.x - center.x) / base_scale;
	    let dy = (center.y - mouse_pos.y) / base_scale;
	    let world_dir = inv_view * Vector3::new(dx, dy, 0.0);
	
	    let target = if target_is_purple { &mut self.selected_vector_purple } else { &mut self.selected_vector };
	    *target = Vector3::new(world_dir.x, world_dir.y, 0.0);
	}


	fn generate_random_matrix(&mut self, range: f32, step: f32) {
        let mut rng = rand::thread_rng();

		let max = (range / step) as i32;

        for r in 0..3 {
            for c in 0..3 {
                // Generates values between -6 and 6, then multiplies by 0.5 
                // to get steps of 0.5 between -3 and 3
                let rand_val = rng.gen_range(-max..=max);
                self.input[r][c] = rand_val as f32 * step;
            }
        }
        self.recalculate_target();
    }
}


impl eframe::App for MatrixApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_hotkeys(ctx);
        let dt = ctx.input(|i| i.unstable_dt).max(1e-6);

		let x_col = egui::Color32::from_hex("#83B366").unwrap();
		let y_col = egui::Color32::from_hex("#FF7154").unwrap();
		let z_col = egui::Color32::from_hex("#8BC9D7").unwrap();

        // --- SIDEBAR ---
        egui::SidePanel::left("controls")
			.width_range(300.0..=350.0)
			.show(ctx, |ui| {
				egui::ScrollArea::vertical()
            	.auto_shrink([false; 2])
            	.show(ui, |ui| {
            		ui.heading("Matrix Visualizer");
            		ui.add_space(4.0);

            		ui.collapsing("⌨ Hotkeys", |ui| {
            		    ui.label("P: Planes | V: Persp | A: Apply\nCtrl+Z: Undo | C: Clear");
            		});
				
            		ui.separator();
            		ui.checkbox(&mut self.perspective, "🔭 Perspective [V]");
				
					ui.separator();
					ui.heading("Feature Toggles");
					ui.checkbox(&mut self.draw_planes, "🔳 Show Original [P]");
					ui.checkbox(&mut self.draw_determinant, "🧊 Determinant [D]");
					ui.checkbox(&mut self.draw_eigen_rays, "✨ Eigen Rays");
					ui.checkbox(&mut self.draw_flow_field, "🌊 Flow Field");
					ui.checkbox(&mut self.draw_unit_sphere, "⚪ Unit Sphere Deformation");

					
					ui.checkbox(&mut self.draw_yellow, egui::RichText::new("Yellow Vector").color(egui::Color32::YELLOW).strong());
					ui.checkbox(&mut self.draw_purple, egui::RichText::new("Purple Vector").color(egui::Color32::from_rgb(160, 32, 240)).strong());
					if self.draw_yellow && self.draw_purple {
						ui.checkbox(&mut self.draw_cross_vector, "X Cross product vector");
						ui.checkbox(&mut self.draw_parallelogram, "▱ Parallellogram");
					}
					if !(self.draw_yellow && self.draw_purple) {
						self.draw_cross_vector = false;
						self.draw_parallelogram = false;
					}
				
					ui.add_space(6.0);
            		ui.add(egui::Slider::new(&mut self.grid_opacity, 0..=255).text("Grid Alpha"));
					ui.add(
					    egui::Slider::new(&mut self.anim_speed, 0.1..=3.0)
					        .text("Animation Speed")
					);
				
            		ui.add_space(10.0);
            		self.draw_matrix_input_ui(ui); // Bryt ut matris-gridden hit
				
            		ui.separator();
            		ui.heading("Resulting Matrix");
				
					egui::Frame::group(ui.style()).show(ui, |ui| {
					    ui.horizontal(|ui| {
						
					        // --- M_total ---
					        ui.vertical(|ui| {
					            ui.label("M");
					            Self::draw_matrix(ui, &self.target);
					        });
						
					        ui.separator();
						
					        // --- Invers ---
					        ui.vertical(|ui| {
					            ui.horizontal(|ui| {
								    ui.label("M");
								    ui.label(
								        egui::RichText::new("-1")
								            .small()
								            .raised()
								    );
								});
							
					            if let Some(inv) = self.target.try_inverse() {
					                Self::draw_matrix(ui, &inv);
					            } else {
					                ui.colored_label(
					                    egui::Color32::LIGHT_RED,
					                    "Not invertible"
					                );
					            }
					        });
					    });
					});
				
					ui.add_space(4.0);
					let det = self.target.determinant();
					let rank = matrix_rank_approx(&self.target, 1e-6);

					ui.label(format!("det(M) = {:.4}", det));
					ui.label(format!("rank(M) = {}", rank));
					if real_eigenpairs_approx(&self.current, 1e-3).is_empty() {
					    ui.colored_label(
					        egui::Color32::GRAY,
					        "No real eigenvectors (complex eigenvalues)"
					    );
					}



					if self.draw_yellow {
						ui.add_space(10.0);
						ui.separator();
						ui.horizontal(|ui| {
							ui.label(
								egui::RichText::new("Yellow")
									.color(egui::Color32::YELLOW)
									.strong()
									.size(18.0)
							
							);
							ui.label(
								egui::RichText::new(" Vector")
									.strong()
									.size(18.0)
							
							);
						});
						ui.horizontal(|ui| {
							for i in 0..3 {
								let label = ["X", "Y", "Z"][i];
								ui.label(format!("{}:", label));
								let id = ui.make_persistent_id(format!("custom_vec_{}", i));
								Self::handle_buffered_input(ui, id, &mut self.input_buffer, &mut self.selected_vector[i]);
							}
						});
						ui.label("[SHIFT + SPACE] to place vector on XY plane");
					
						// --- Transformed Yellow Vector (framed) ---
						let y_t = self.current * self.selected_vector;
					
						ui.add_space(6.0);
						egui::Frame::group(ui.style()).show(ui, |ui| {
						    ui.label(
						        egui::RichText::new("Transformed (M·v)")
						            .color(egui::Color32::YELLOW)
						            .strong()
						    );
						
						    ui.horizontal(|ui| {
						        for i in 0..3 {
						            let val = y_t[i];
						            let color = if val.abs() < 0.001 {
						                egui::Color32::DARK_GRAY
						            } else if val > 0.0 {
						                egui::Color32::LIGHT_GREEN
						            } else {
						                egui::Color32::LIGHT_RED
						            };
						            ui.colored_label(color, format!("{:>6.3}", val));
						        }
						    });
						});
					
					
					}
				
					if self.draw_purple {
						ui.add_space(10.0);
						ui.separator();
						ui.horizontal(|ui| {
							ui.label(
								egui::RichText::new("Purple")
									.color(egui::Color32::from_rgb(160, 32, 240))
									.strong()
									.size(18.0)
							);
							ui.label(
								egui::RichText::new(" Vector")
									.strong()
									.size(18.0)
							);
						});
					
						ui.horizontal(|ui| {
							for i in 0..3 {
								let id = ui.make_persistent_id(format!("purple_vec_{}", i));
								Self::handle_buffered_input(ui, id, &mut self.input_buffer, &mut self.selected_vector_purple[i]);
							}
						});
						ui.label("[SHIFT + SPACE] to place vector on XY plane");
					
						// --- Transformed Purple Vector (framed) ---
						let p_t = self.current * self.selected_vector_purple;

						ui.add_space(6.0);
						egui::Frame::group(ui.style()).show(ui, |ui| {
						    ui.label(
						        egui::RichText::new("Transformed (M·v)")
						            .color(egui::Color32::from_rgb(160, 32, 240))
						            .strong()
						    );
						
						    ui.horizontal(|ui| {
						        for i in 0..3 {
						            let val = p_t[i];
						            let color = if val.abs() < 0.001 {
						                egui::Color32::DARK_GRAY
						            } else if val > 0.0 {
						                egui::Color32::LIGHT_GREEN
						            } else {
						                egui::Color32::LIGHT_RED
						            };
						            ui.colored_label(color, format!("{:>6.3}", val));
						        }
						    });
						});
					

					}

					if self.draw_yellow && self.draw_purple {
					
						ui.add_space(10.0);
						ui.separator();
						ui.heading("Vector Properties");
						let btn_text = if self.cross_reverse_order { "Lila × Gul" } else { "Gul × Lila" };
						if ui.button(format!("Byt ordning: {}", btn_text)).clicked() {
							self.cross_reverse_order = !self.cross_reverse_order;
						}
						// --- Cross product coordinates ---
						let m = self.current;
						let yellow_t = m * self.selected_vector;
						let purple_t = m * self.selected_vector_purple;
					
						let cross = if self.cross_reverse_order {
							purple_t.cross(&yellow_t)
						} else {
							yellow_t.cross(&purple_t)
						};
					
						ui.add_space(6.0);
						egui::Frame::group(ui.style()).show(ui, |ui| {
							ui.horizontal(|ui| {
							
								// --- Crossproduct (column 1) ---
								ui.vertical(|ui| {
									ui.label("Cross product:");
								
									for i in 0..3 {
										let val = cross[i];
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
							
								ui.separator();
							
								// --- Dot-product (column 2) ---
								ui.vertical(|ui| {
									ui.label("Dot product:");
								
									let dot = yellow_t.dot(&purple_t);
								
									let color = if dot.abs() < 0.001 {
										egui::Color32::DARK_GRAY
									} else if dot > 0.0 {
										egui::Color32::LIGHT_GREEN
									} else {
										egui::Color32::LIGHT_RED
									};
								
									ui.colored_label(color, format!("{:.3}", dot));
								});
							});
						});
					}
				});

        });

        // --- VIEWPORT ---
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
			
            
            // Input handling
            if resp.dragged_by(egui::PointerButton::Primary) {
                self.view_rot += resp.drag_delta().x * 0.01;
                self.view_pitch = (self.view_pitch - resp.drag_delta().y * -0.01).clamp(-1.5, 1.5);
            }
            self.view_zoom = (self.view_zoom * (1.0 + ui.input(|i| i.smooth_scroll_delta.y) * 0.001)).clamp(0.1, 10.0);

            // 1. Lyft ut variabler som closuren behöver för att undvika att låsa hela 'self'
			let perspective_mode = self.perspective; 
			let view_zoom = self.view_zoom;

			// Setup Projection
			let painter = ui.painter_at(rect);
			let view_mat = self.get_view_matrix();
			let base_scale = (rect.width().min(rect.height()) / 25.0) * view_zoom;

			// Nu lånar closuren bara 'perspective_mode' och 'base_scale', inte 'self'
			let project = |v: Vector3<f32>| {
			    let v_v = view_mat * v;
			    let factor = if perspective_mode { 
			        (base_scale * 20.0) / (20.0 - v_v.z).max(0.1) 
			    } else { 
			        base_scale 
			    };
			    rect.center() + egui::vec2(v_v.x * factor, -v_v.y * factor)
			};

			let is_space = ctx.input(|i| i.key_down(egui::Key::Space));
			let is_shift = ctx.input(|i| i.modifiers.shift);

			if is_space {
			    if let Some(pos) = ctx.pointer_interact_pos() {
			        if rect.contains(pos) {
			            // Om shift är nere -> Lila, annars -> Gul
			            self.place_vector_at_mouse(pos, rect, view_mat, is_shift);
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

            // Render Scene

			if self.draw_flow_field {
			    draw_flow_field(
			        &painter,
			        &project,
			        &self.current,
			        0.8, // spacing
			        6,   // extent
			    );
			}

            if self.draw_planes && !self.current.is_identity(1e-10)  { draw_origin_planes(&painter, &project, self.grid_size); }
            
            let grid_c = egui::Color32::from_rgba_unmultiplied(80, 140, 220, self.grid_opacity);
            draw_grid_3d(&painter, &project, &self.current, grid_c, self.grid_size);
            
			if self.draw_determinant {
			    draw_determinant_geometry(
			        &painter,
			        &project,
			        &self.current, 
			    );
			}

			if self.draw_unit_sphere
			    && matrix_rank_approx(&self.current, 1e-6) >= 2
			{
			    //draw_deformed_sphere(&painter, &project, &self.current);
			    draw_colored_unit_sphere(&painter, &project, &self.current);
				
			}

			if self.draw_eigen_rays && !is_near_identity(&self.current, 1e-5) {
			    draw_eigen_rays(&painter, &project, &self.current);
			}



            
            draw_axes_3d(&painter, &project);
            
            // Render Basis Vectors
            let m = self.current;
            let vectors = [
				(m*Vector3::x(), x_col),
				(m*Vector3::y(), y_col), 
				(m*Vector3::z(), z_col),
			];
            
            for (v, color) in vectors {
                draw_arrow(&painter, project(Vector3::zeros()), project(v), color);
            }

			// Render Basis Vectors & Custom Vectors
			let m = self.current;
			let yellow_transformed = m * self.selected_vector;
			let purple_transformed = m * self.selected_vector_purple;
				
			// Beräkna kryssprodukt baserat på ordning
			let cross_prod = if self.cross_reverse_order {
				purple_transformed.cross(&yellow_transformed)
			} else {
				yellow_transformed.cross(&purple_transformed)
			};

			// --- Parallelogram between yellow & purple ---
			let m = self.current;
			let y = m * self.selected_vector;
			let p = m * self.selected_vector_purple;
			
			if self.draw_parallelogram {
				let poly = vec![
				    project(Vector3::zeros()),
				    project(y),
				    project(y + p),
				    project(p),
				];
				
				painter.add(egui::Shape::convex_polygon(
				    poly,
				    egui::Color32::from_rgba_unmultiplied(180, 180, 180, 40),
				    egui::Stroke::new(1.0, egui::Color32::WHITE),
				));
			}
			
			// Rita pilarna
			if self.draw_yellow { draw_arrow(&painter, project(Vector3::zeros()), project(yellow_transformed), egui::Color32::YELLOW); }
			if self.draw_purple { draw_arrow(&painter, project(Vector3::zeros()), project(purple_transformed), egui::Color32::from_rgb(160, 32, 240)); } // Lila
			
			
			// Rita kryssprodukten (t.ex. vit eller cyan för att synas bra)
			if self.draw_cross_vector && cross_prod.norm() > 0.001 {
				draw_arrow(&painter, project(Vector3::zeros()), project(cross_prod), egui::Color32::WHITE);
			}

            // UI Elements (Nav Cube)
            draw_nav_cube(ui, &painter, &view_mat, self);
        });

        ctx.request_repaint();
    }
}


// --- Helpers ---

fn draw_unit_cube(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2, m: &Matrix3<f32>, draw_volume: bool,) {
    let det = m.determinant();
    let abs_det = det.abs();
    
    // Color based on orientation (Right-handed vs Left-handed system)
    let color = if det >= 0.0 {
        egui::Color32::from_rgba_unmultiplied(120, 80, 200, 40)
    } else {
        egui::Color32::from_rgba_unmultiplied(200, 80, 80, 40)
    };
    
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(200, 150, 255, 180));
    
    let corners = [
        Vector3::new(0.,0.,0.), Vector3::new(1.,0.,0.), Vector3::new(1.,1.,0.), Vector3::new(0.,1.,0.),
        Vector3::new(0.,0.,1.), Vector3::new(1.,0.,1.), Vector3::new(1.,1.,1.), Vector3::new(0.,1.,1.),
    ].map(|v| project(m * v));

    let faces = [[0,1,2,3], [4,5,6,7], [0,4,7,3], [1,5,6,2], [0,1,5,4], [3,2,6,7]];

    for face in faces {
        painter.add(egui::Shape::convex_polygon(face.iter().map(|&i| corners[i]).collect(), color, egui::Stroke::NONE));
    }
    
    let edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for edge in edges {
        painter.line_segment([corners[edge[0]], corners[edge[1]]], stroke);
    }

    // --- DISPLAY VOLUME LABEL ---
	if draw_volume && abs_det > 0.001 {
	    let center_world = m * Vector3::new(0.5, 0.5, 0.5);
	    let center_screen = project(center_world);
	
	    painter.text(
			center_screen,
	        egui::Align2::CENTER_CENTER,
	        format!("Vol: {:.2}", abs_det),
	        egui::FontId::proportional(14.0), // Changed from .bold()
	        egui::Color32::WHITE
	    );
	}
}


fn draw_determinant_geometry(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    m: &Matrix3<f32>,
) {
    let eps = 1e-6;
    match matrix_rank_approx(m, eps) {
        3 => {
            draw_unit_cube(painter, project, m, true);
        }
        2 => {
            let (a, b) = best_parallelogram_basis(m);
            let poly = vec![
                project(Vector3::zeros()),
                project(a),
                project(a + b),
                project(b),
            ];
			// Kamera-normal i world space
			let view_normal = Vector3::new(0.0, 0.0, 1.0);

			// Orientering (signerad area)
			let signed_area = a.cross(&b).dot(&view_normal);

			let color = if signed_area >= 0.0 {
			    egui::Color32::from_rgba_unmultiplied(120, 200, 120, 60) // grön
			} else {
			    egui::Color32::from_rgba_unmultiplied(200, 80, 80, 60)   // röd
			};

            painter.add(egui::Shape::convex_polygon(
                poly,
                color,
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            ));
        }
        1 => {
            let v = m.column(0).into_owned();
            painter.line_segment(
                [project(Vector3::zeros()), project(v)],
                egui::Stroke::new(2.5, egui::Color32::LIGHT_RED),
            );
        }
        _ => {}
    }
}





fn draw_grid_3d(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2, m: &Matrix3<f32>, color: egui::Color32, size: i32) {
    // Vi skapar en stroke baserat på färgen vi skickar in
    let stroke = egui::Stroke::new(1.0, color); 
    let s = size as f32;
    
    for i in -size..=size {
        let t = i as f32;
        // XY Plan
        painter.line_segment([project(m * Vector3::new(t, -s, 0.0)), project(m * Vector3::new(t, s, 0.0))], stroke);
        painter.line_segment([project(m * Vector3::new(-s, t, 0.0)), project(m * Vector3::new(s, t, 0.0))], stroke);
        
        // XZ Plan
        painter.line_segment([project(m * Vector3::new(t, 0.0, -s)), project(m * Vector3::new(t, 0.0, s))], stroke);
        painter.line_segment([project(m * Vector3::new(-s, 0.0, t)), project(m * Vector3::new(s, 0.0, t))], stroke);
        
        // YZ Plan
        painter.line_segment([project(m * Vector3::new(0.0, t, -s)), project(m * Vector3::new(0.0, t, s))], stroke);
        painter.line_segment([project(m * Vector3::new(0.0, -s, t)), project(m * Vector3::new(0.0, s, t))], stroke);
    }
}


fn draw_axes_3d(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2) {
    let s = egui::Stroke::new(1.0, egui::Color32::GRAY.linear_multiply(0.2));
    for i in 0..3 {
        let mut start = Vector3::zeros(); let mut end = Vector3::zeros();
        start[i] = -10.0; end[i] = 10.0;
        painter.line_segment([project(start), project(end)], s);
    }
}


fn smoothstep(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }


fn draw_arrow(painter: &egui::Painter, start: egui::Pos2, end: egui::Pos2, color: egui::Color32) {
    let vec = end - start;
    let len = vec.length();
    if len < 1.0 { return; }
    
    // Main shaft
    painter.line_segment([start, end], egui::Stroke::new(2.5, color));
    
    // Arrow head (triangle)
    let head_len = (len * 0.15).clamp(5.0, 15.0);
    let dir = vec / len;
    let perp = egui::vec2(-dir.y, dir.x) * (head_len * 0.4);
    
    let tip = end;
    let base = end - dir * head_len;
    
    painter.add(egui::Shape::convex_polygon(
        vec![tip, base + perp, base - perp],
        color,
        egui::Stroke::NONE,
    ));
}


fn draw_origin_planes(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2, size: i32) {
    let identity = nalgebra::Matrix3::identity();
    // Vi använder en diskret grå färg med låg opacitet för det statiska rutnätet
    let color = egui::Color32::from_rgba_unmultiplied(150, 150, 150, 40);
    
    draw_grid_3d(
        painter, 
        project, 
        &identity, 
        color, 
        size
    );
}


fn draw_nav_cube(ui: &egui::Ui, painter: &egui::Painter, view_mat: &Matrix3<f32>, app: &mut MatrixApp) {
    let rect = painter.clip_rect();
    let nav_center = egui::pos2(rect.right() - 60.0, rect.top() + 60.0);
    let nav_scale = 30.0;

    let faces = [
        ("XY ", Vector3::new(0.0, 0.0, 1.0), 0.0, 0.0),
        ("-XY", Vector3::new(0.0, 0.0, -1.0), std::f32::consts::PI, 0.0),
        ("YZ ", Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2, 0.0),
        ("-YZ", Vector3::new(-1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2, 0.0),
        ("XZ ", Vector3::new(0.0, 1.0, 0.0), 0.0, std::f32::consts::FRAC_PI_2),
        ("-XZ", Vector3::new(0.0, -1.0, 0.0), 0.0, -std::f32::consts::FRAC_PI_2),
    ];

    let project_nav = |v: Vector3<f32>| {
        let v_view = view_mat * v;
        egui::pos2(nav_center.x + v_view.x * nav_scale, nav_center.y - v_view.y * nav_scale)
    };

    let mut sorted_faces: Vec<_> = faces.iter().collect();
    sorted_faces.sort_by(|a, b| {
        let az = (view_mat * a.1).z;
        let bz = (view_mat * b.1).z;
        az.partial_cmp(&bz).unwrap()
    });

    for (name, normal, yaw, pitch) in sorted_faces {
        let view_normal = view_mat * normal;
        if view_normal.z <= 0.0 { continue; }

        let (u, v) = if normal.x.abs() > 0.9 {
            (Vector3::y(), Vector3::z())
        } else {
            (Vector3::x(), normal.cross(&Vector3::x()).normalize())
        };

        let corners = [
            project_nav(normal + u + v),
            project_nav(normal - u + v),
            project_nav(normal - u - v),
            project_nav(normal + u - v),
        ];

        let is_hovered = ui.rect_contains_pointer(egui::Rect::from_points(&corners));
        let color = if is_hovered { egui::Color32::from_rgb(200, 100, 0) } else { egui::Color32::from_gray(60) };
        
        painter.add(egui::Shape::convex_polygon(corners.to_vec(), color, egui::Stroke::new(1.0, egui::Color32::WHITE)));
        painter.text(project_nav(*normal), egui::Align2::CENTER_CENTER, &name[0..3], egui::FontId::proportional(12.0), egui::Color32::WHITE);

        if is_hovered && ui.input(|i| i.pointer.any_click()) {
            app.set_view(*yaw, *pitch);
        }
    }

    // --- DRAW BASIS EDGES (With Occlusion Check) ---
    let origin_v = Vector3::new(-1.0, -1.0, -1.0);
    let axes_v = [
        Vector3::new(1.0, -1.0, -1.0), // X
        Vector3::new(-1.0, 1.0, -1.0), // Y
        Vector3::new(-1.0, -1.0, 1.0), // Z
    ];

    let cols = [
        egui::Color32::from_hex("#83B366").unwrap(),
        egui::Color32::from_hex("#FF7154").unwrap(),
        egui::Color32::from_hex("#8BC9D7").unwrap(),
    ];

    for i in 0..3 {
        let end_v = axes_v[i];
        // Calculate midpoint in world space
        let midpoint = (origin_v + end_v) * 0.5;
        // Transform midpoint to view space to check depth
        let midpoint_view = view_mat * midpoint;

        // If the midpoint's Z is > 0, the edge is on the side facing the camera
        if midpoint_view.z > 0.0 {
            let p1 = project_nav(origin_v);
            let p2 = project_nav(end_v);
            painter.line_segment([p1, p2], egui::Stroke::new(3.0, cols[i]));
        }
    }
}

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Ladda emoji-font (fallback)
    fonts.font_data.insert(
        "emoji".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../assets/fonts/NotoEmoji-Regular.ttf"
        )),
    );

    // Lägg den SIST i proportional-fonten → fallback
    fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .unwrap()
        .push("emoji".to_owned());

    ctx.set_fonts(fonts);
}


fn draw_eigen_rays(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    m: &Matrix3<f32>,
) {
    let rays = real_eigenpairs_approx(m, 1e-3);

    for (v, lambda) in rays {
        let dir = v.normalize() * 10.0;
        let color = if lambda >= 0.0 {
            egui::Color32::from_rgb(120, 160, 255) // blå
        } else {
            egui::Color32::from_rgb(220, 80, 80)   // röd
        };

        let stroke = egui::Stroke::new(
            (lambda.abs() * 2.0).clamp(1.5, 4.0),
            color,
        );

        painter.line_segment(
            [project(-dir), project(dir)],
            stroke,
        );
    }
}


fn draw_flow_field(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    m: &Matrix3<f32>,
    spacing: f32,
    extent: i32,
) {
    for ix in -extent..=extent {
        for iy in -extent..=extent {
            let v = Vector3::new(ix as f32 * spacing, iy as f32 * spacing, 0.0);
            let mv = m * v;

            let dir = mv - v;
            if dir.norm() < 0.01 { continue; }

            let end = v + dir.normalize() * spacing * 0.8;

            painter.line_segment(
                [project(v), project(end)],
                egui::Stroke::new(1.2, egui::Color32::from_rgba_unmultiplied(200, 200, 200, 120)),
            );
        }
    }
}


fn draw_deformed_sphere(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    m: &Matrix3<f32>,
) {
    let sphere = sample_unit_sphere(16, 32);

    let stroke = egui::Stroke::new(
        1.0,
        egui::Color32::from_rgba_unmultiplied(200, 200, 255, 80),
    );

    let rows = 17; // n_theta + 1
    let cols = 33; // n_phi + 1

    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            let p = m * sphere[idx];

            // longitud
            if j + 1 < cols {
                let q = m * sphere[idx + 1];
                painter.line_segment([project(p), project(q)], stroke);
            }

            // latitud
            if i + 1 < rows {
                let q = m * sphere[idx + cols];
                painter.line_segment([project(p), project(q)], stroke);
            }
        }
    }
}


fn draw_colored_unit_sphere(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    m: &Matrix3<f32>,
) {
    let points = sample_unit_sphere(18, 36);

    for p in points {
        let p_hat = p.normalize();
        let mp = m * p_hat;
        let delta = mp - p_hat;

        let radial = delta.dot(&p_hat).abs();
        let tangent = (delta - p_hat * delta.dot(&p_hat)).norm();

        let t = (radial / (radial + tangent + 1e-6)).clamp(0.0, 1.0);

        // röd → gul → grön
        let color = egui::Color32::from_rgb(
            (255.0 * (1.0 - t.powi(2))) as u8,
            (255.0 * t.powi(2)) as u8,
            80,
        );

        let screen_pos = project(mp);
        painter.circle_filled(screen_pos, 2.0, color);
    }
}
