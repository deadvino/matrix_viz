use eframe::egui;
use nalgebra::{Matrix3, Vector3};
use rand::Rng; // This must be present to use .gen_range()


use crate::math::is_near_identity;
use crate::math::real_eigenpairs_approx;
use crate::math::matrix_rank_approx;


use crate::render::draw_grid_3d;
use crate::render::draw_axes_3d;
use crate::render::draw_arrow;
use crate::render::draw_origin_planes;
use crate::render::draw_nav_cube;
use crate::render::draw_determinant_geometry;
use crate::render::draw_eigen_rays;
use crate::render::draw_colored_unit_sphere;
use crate::render::draw_flow_field;



pub struct MatrixApp {
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


	pub fn set_view(&mut self, yaw: f32, pitch: f32) {
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

fn smoothstep(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }


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