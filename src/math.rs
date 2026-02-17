use nalgebra::{Vector3, Matrix3};


pub fn sample_unit_sphere(n_theta: usize, n_phi: usize) -> Vec<Vector3<f32>> {
    let mut pts = Vec::new();

    for i in 0..=n_theta {
        let theta = std::f32::consts::PI * i as f32 / n_theta as f32;
        for j in 0..=n_phi {
            let phi = 2.0 * std::f32::consts::PI * j as f32 / n_phi as f32;

            pts.push(Vector3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ));
        }
    }
    pts
}


pub fn is_near_identity(m: &Matrix3<f32>, eps: f32) -> bool {
    (m - Matrix3::identity()).norm() < eps
}


pub fn snap(value: f32, step: f32) -> f32 {
    (value / step).round() * step
}



pub fn real_eigenpairs_exact(m: &Matrix3<f32>, eps: f32) -> Vec<(Vector3<f32>, f32)> {
    let mut result = Vec::new();
    let values = m.complex_eigenvalues();
    
    // 1. Isolate unique real eigenvalues
    let mut unique_lambdas: Vec<f32> = Vec::new();
    for l in values.iter() {
        if l.im.abs() < eps {
            if !unique_lambdas.iter().any(|&already| (already - l.re).abs() < eps) {
                unique_lambdas.push(l.re);
            }
        }
    }

    // 2. Find all vectors for every unique eigenvalue
    for lambda in unique_lambdas {
        let mut system = *m;
        system[(0,0)] -= lambda;
        system[(1,1)] -= lambda;
        system[(2,2)] -= lambda;
        
        let svd = system.svd(false, true);
        let v_t = svd.v_t.unwrap();

        for j in 0..3 {
            // If singular value is close to zero -> found an eigenvector
            if svd.singular_values[j].abs() < eps * 10.0 {
                let v = v_t.row(j).transpose().normalize();
                
                // Remove doubles
				if !result.iter().any(|(old_v, old_l): &(Vector3<f32>, f32)| {
				    let same_lambda = (old_l - lambda).abs() < eps;
				    // .abs() on dot-product catches both v and -v
				    let parallel = old_v.dot(&v).abs() > (1.0 - eps);
				
				    same_lambda && parallel
				}) {
				    result.push((v, lambda));
				}
            }
        }
    }
    result
}


pub fn best_parallelogram_basis(m: &Matrix3<f32>) -> (Vector3<f32>, Vector3<f32>) {
	let v = m.column(0).into_owned();
	let u = m.column(1).into_owned();
	let w = m.column(2).into_owned();

	let vu = v.cross(&u).norm_squared();
	let vw = v.cross(&w).norm_squared();
	let uw = u.cross(&w).norm_squared();

	if vu >= vw && vu >= uw {
		(v, u)
	} else if vw >= uw {
		(v, w)
	} else {
		(u, w)
	}
}


pub fn matrix_rank_approx(m: &Matrix3<f32>, eps: f32) -> usize {
    let v = m.column(0);
    let u = m.column(1);
    let w = m.column(2);

    let vu = v.cross(&u).norm_squared();
    let vw = v.cross(&w).norm_squared();
    let uw = u.cross(&w).norm_squared();

    if vu > eps && vw > eps && uw > eps {
        3
    } else if vu > eps || vw > eps || uw > eps {
        2
    } else if v.norm_squared() > eps || u.norm_squared() > eps || w.norm_squared() > eps {
        1
    } else {
        0
    }
}


pub fn parse_plane_equation(equation: &str) -> Option<(f32, f32, f32, f32)> {
    let equation = equation.replace(" ", "");
    let parts: Vec<&str> = equation.split('=').collect();

    if parts.len() != 2 || parts[1] != "0" {
        return None;
    }

    let mut a = 0.0;
    let mut b = 0.0;
    let mut c = 0.0;
    let mut d = 0.0;

    // Split terms
    let terms = parts[0].split('+');
    for term in terms {
        if term.contains('x') {
            let coeff = term.replace("x", "");
            a = if coeff.is_empty() { 1.0 } else { coeff.parse().unwrap_or(0.0) };
        } else if term.contains('y') {
            let coeff = term.replace("y", "");
            b = if coeff.is_empty() { 1.0 } else { coeff.parse().unwrap_or(0.0) };
        } else if term.contains('z') {
            let coeff = term.replace("z", "");
            c = if coeff.is_empty() { 1.0 } else { coeff.parse().unwrap_or(0.0) };
        } else if !term.is_empty() {
            d = term.parse().unwrap_or(0.0);
        }
    }

    Some((a, b, c, d))
}


pub fn parse_line_equation(equation: &str) -> Option<(Vector3<f32>, Vector3<f32>)> {
    let equation = equation.replace(" ", "");
    let parts: Vec<&str> = equation.split('+').collect();

    if parts.len() != 2 {
        return None;
    }

    // Parse point
    let point_str = parts[0].trim_start_matches("(x,y,z)=").trim_matches(|c| c == '(' || c == ')');
    let point_coords: Vec<f32> = point_str.split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    if point_coords.len() != 3 {
        return None;
    }
    let point = Vector3::new(point_coords[0], point_coords[1], point_coords[2]);

    // Parse direction
    let dir_str = parts[1].trim_start_matches('t').trim_matches(|c| c == '(' || c == ')');
    let dir_coords: Vec<f32> = dir_str.split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    if dir_coords.len() != 3 {
        return None;
    }
    let direction = Vector3::new(dir_coords[0], dir_coords[1], dir_coords[2]);

    Some((point, direction))
}


pub fn line_plane_intersection(
    line_point: Vector3<f32>,
    line_dir: Vector3<f32>,
    plane_normal: Vector3<f32>,
    plane_d: f32
) -> Option<Vector3<f32>> {
    let denom = plane_normal.dot(&line_dir);
    if denom.abs() < 1e-6 {
        return None; // Line is parallel to plane
    }

    let t = -(plane_normal.dot(&line_point) + plane_d) / denom;
    Some(line_point + line_dir * t)
}


pub fn line_line_intersection(
    p1: Vector3<f32>,
    d1: Vector3<f32>,
    p2: Vector3<f32>,
    d2: Vector3<f32>
) -> Option<Vector3<f32>> {
    let d1_cross_d2 = d1.cross(&d2);
    if d1_cross_d2.norm() < 1e-6 {
        return None; // Lines are parallel
    }

    let p2_minus_p1 = p2 - p1;
    let t = p2_minus_p1.cross(&d2).dot(&d1_cross_d2) / d1_cross_d2.norm_squared();
    Some(p1 + d1 * t)
}
