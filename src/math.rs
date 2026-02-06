use nalgebra::{Vector3, Matrix3};
use nalgebra::RealField;
use nalgebra::ComplexField;

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


pub fn real_eigenpairs_approx(m: &Matrix3<f32>, eps: f32) -> Vec<(Vector3<f32>, f32)> {
    let mut result = Vec::new();

    let dirs = [
        Vector3::x(),
        Vector3::y(),
        Vector3::z(),
        Vector3::new(1.0, 1.0, 0.0).normalize(),
        Vector3::new(1.0, 0.0, 1.0).normalize(),
        Vector3::new(0.0, 1.0, 1.0).normalize(),
    ];

    for v in dirs {
        let mv = m * v;

        let denom = v.dot(&v);
        if denom.abs() < eps {
            continue;
        }

        let lambda = mv.dot(&v) / denom;

        if (mv - v * lambda).norm() < eps {
            result.push((v, lambda));
        }
    }

    result
}



pub fn real_eigenpairs_exact(m: &Matrix3<f32>, eps: f32) -> Vec<(Vector3<f32>, f32)> {
    let mut result = Vec::new();

    // Vi tvingar nalgebra att använda den generella dekomponeringen 
    // genom att använda .eigen() på en ägd matris.
    // Om .eigen() inte hittas, använd .complex_eigenvalues() 
    // i kombination med att vi löser (M - λI)v = 0 manuellt för varje λ.
    
    // 1. Hitta egenvärden (denna metod finns nästan alltid)
	let values = m.complex_eigenvalues();
	
	for lambda_complex in values.iter() {
	    if lambda_complex.im.abs() < eps {
	        let lambda = lambda_complex.re;
		
	        // 2. Lös (M - λI)v = 0 för att hitta egenvektorn
	        // Vi skapar matrisen (M - λI)
	        let mut system = *m;
	        system[(0,0)] -= lambda;
	        system[(1,1)] -= lambda;
	        system[(2,2)] -= lambda;
		
	        // 3. Använd SVD för att hitta nollrummet (kernel)
	        // Vektorn som motsvarar det minsta singulärvärdet är vår egenvektor.
	        let svd = system.svd(false, true);
	        if let Some(v_matrix) = svd.v_t {
	            // Sista raden i V^T (eller sista kolumnen i V) är egenvektorn
	            let v = v_matrix.row(2).transpose().normalize();
	            result.push((v, lambda));
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