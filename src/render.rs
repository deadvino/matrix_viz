//use egui::Painter;
use nalgebra::{Matrix3, Vector3};

use crate::math::matrix_rank_approx;
use crate::math::best_parallelogram_basis;
use crate::math::real_eigenpairs_exact;
use crate::math::sample_unit_sphere;

use crate::app::MatrixApp;


pub fn draw_grid_3d(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2, m: &Matrix3<f32>, color: egui::Color32, size: i32) {
    let stroke = egui::Stroke::new(1.0, color); 
    let s = size as f32;
    
    for i in -size..=size {
        let t = i as f32;
        // XY Plane
        painter.line_segment([project(m * Vector3::new(t, -s, 0.0)), project(m * Vector3::new(t, s, 0.0))], stroke);
        painter.line_segment([project(m * Vector3::new(-s, t, 0.0)), project(m * Vector3::new(s, t, 0.0))], stroke);
        
        // XZ Plane
        painter.line_segment([project(m * Vector3::new(t, 0.0, -s)), project(m * Vector3::new(t, 0.0, s))], stroke);
        painter.line_segment([project(m * Vector3::new(-s, 0.0, t)), project(m * Vector3::new(s, 0.0, t))], stroke);
        
        // YZ Plane
        painter.line_segment([project(m * Vector3::new(0.0, t, -s)), project(m * Vector3::new(0.0, t, s))], stroke);
        painter.line_segment([project(m * Vector3::new(0.0, -s, t)), project(m * Vector3::new(0.0, s, t))], stroke);
    }
}


pub fn draw_axes_3d(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2) {
    let s = egui::Stroke::new(1.0, egui::Color32::GRAY.linear_multiply(0.2));
    for i in 0..3 {
        let mut start = Vector3::zeros(); let mut end = Vector3::zeros();
        start[i] = -10.0; end[i] = 10.0;
        painter.line_segment([project(start), project(end)], s);
    }
}


pub fn draw_arrow(painter: &egui::Painter, start: egui::Pos2, end: egui::Pos2, color: egui::Color32) {
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


pub fn draw_origin_planes(painter: &egui::Painter, project: &impl Fn(Vector3<f32>) -> egui::Pos2, size: i32) {
    let identity = nalgebra::Matrix3::identity();
    // Nice and quiet gray colour for static grid
    let color = egui::Color32::from_rgba_unmultiplied(150, 150, 150, 40);
    
    draw_grid_3d(
        painter, 
        project, 
        &identity, 
        color, 
        size
    );
}


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
	        egui::FontId::proportional(14.0),
	        egui::Color32::WHITE
	    );
	}
}


pub fn draw_determinant_geometry(
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
			// Camera-normal in world space
			let view_normal = Vector3::new(0.0, 0.0, 1.0);

			// Orientation (signed area)
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


pub fn draw_eigen_rays(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    m: &Matrix3<f32>,
) {
    let rays = real_eigenpairs_exact(m, 1e-4);

    for (v, lambda) in rays {
        let dir = v.normalize() * 10.0;
        let color = if lambda.abs() <= 0.001 {
		    egui::Color32::from_rgb(100, 150, 255) // Blue
		} else if lambda > 0.001 {
		    egui::Color32::from_rgb(120, 255, 160) // Green
		} else {
		    egui::Color32::from_rgb(220, 80, 80)   // Red
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


pub fn draw_colored_unit_sphere(
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

        // red -> yellow -> green
        let color = egui::Color32::from_rgb(
            (255.0 * (1.0 - t.powi(2))) as u8,
            (255.0 * t.powi(2)) as u8,
            80,
        );

        let screen_pos = project(mp);
        painter.circle_filled(screen_pos, 2.0, color);
    }
}


pub fn draw_flow_field(
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


pub fn draw_nav_cube(ui: &egui::Ui, painter: &egui::Painter, view_mat: &Matrix3<f32>, app: &mut MatrixApp) {
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


pub fn draw_plane(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    size: f32,
    color: egui::Color32,
) {
    let normal = Vector3::new(a, b, c).normalize();
    let point_on_plane = if a != 0.0 {
        Vector3::new(-d/a, 0.0, 0.0)
    } else if b != 0.0 {
        Vector3::new(0.0, -d/b, 0.0)
    } else if c != 0.0 {
        Vector3::new(0.0, 0.0, -d/c)
    } else {
        return;
    };

    let u = if normal.x.abs() > 0.1 {
        Vector3::new(normal.y, -normal.x, 0.0).normalize()
    } else {
        Vector3::new(0.0, normal.z, -normal.y).normalize()
    };
    let v = normal.cross(&u).normalize();

    let stroke = egui::Stroke::new(1.0, color);
    let half_size = size * 0.5;

    for i in -10..=10 {
        let t = i as f32 * 0.1;
        painter.line_segment(
            [project(point_on_plane + u * half_size + v * t * size),
             project(point_on_plane - u * half_size + v * t * size)],
            stroke,
        );
        painter.line_segment(
            [project(point_on_plane + v * half_size + u * t * size),
             project(point_on_plane - v * half_size + u * t * size)],
            stroke,
        );
    }
}


pub fn draw_line(
    painter: &egui::Painter,
    project: &impl Fn(Vector3<f32>) -> egui::Pos2,
    point: Vector3<f32>,
    direction: Vector3<f32>,
    length: f32,
    color: egui::Color32,
) {
    let dir_normalized = direction.normalize();
    let start = point - dir_normalized * length * 0.5;
    let end = point + dir_normalized * length * 0.5;

    painter.line_segment(
        [project(start), project(end)],
        egui::Stroke::new(2.0, color),
    );

    // // Draw arrowhead
    // let arrow_size = length * 0.1;
    // let arrow_dir = dir_normalized * arrow_size;
    // let perp1 = arrow_dir.perp(&Vector3::x()).normalize() * arrow_size * 0.3;
    // let perp2 = arrow_dir.perp(&Vector3::y()).normalize() * arrow_size * 0.3;

    // painter.add(egui::Shape::convex_polygon(
    //     vec![
    //         project(end),
    //         project(end - arrow_dir + perp1),
    //         project(end - arrow_dir + perp2),
    //     ],
    //     color,
    //     egui::Stroke::NONE,
    // ));
}
