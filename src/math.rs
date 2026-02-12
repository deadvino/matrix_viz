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