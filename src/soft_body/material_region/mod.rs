
use super::*;
use std::sync::Arc;

pub use self::region::*;
pub use self::material::*;
pub use self::interaction::*;

mod region;
mod material;
mod interaction;

pub type MaterialBuffer = Buffer<[Material], ReadWrite>;
pub type InteractionBuffer = Buffer<[MatInteraction], ReadWrite>;
pub type Materials = (MaterialBuffer, InteractionBuffer);

#[derive(Clone)]
pub struct MaterialRegion {
    pub region: Arc<dyn Region+Send+Sync>,
    pub packing_coefficient: f32,
    pub mat: Material,
    pub vel: Arc<dyn Fn(vec4)->vec4>
}

impl MaterialRegion {

    pub fn new<R:Region+Send+Sync>(region: R, packing: f32, mat: Material) -> Self {
        MaterialRegion {
            region: Arc::new(region),
            packing_coefficient: packing,
            mat: mat,
            vel: Arc::new(|_| vec4::default())
        }
    }

    pub fn with_vel<R:Region+Send+Sync, V:Fn(vec4)->vec4+'static>(region: R, packing: f32, mat: Material, v:V) -> Self {
        MaterialRegion {
            region: Arc::new(region),
            packing_coefficient: packing,
            mat: mat,
            vel: Arc::new(v)
        }
    }

    pub fn gen_particles(&self, mut h: f32, mat_id: u32) -> (Vec<Particle>, Material) {

        h = self.packing_coefficient * h;

        let bound = self.region.bound();


        let mut num_in_box = 0u64;
        let mut pos = bound.min;
        let start_density = self.mat.start_den;

        use std::thread::*;

        //the max number of particles to test in each thread
        const JOB_SIZE:usize = 8192;

        //the particle testing threads
        let mut threads = Vec::<JoinHandle<Vec<Particle>>>::new();

        //the list accumulating positions to test in the next job
        let mut particles_to_test = Vec::<(vec4, vec4)>::with_capacity(JOB_SIZE);

        //the closure the spawns a new thread for particle testing
        let dispatch: &mut dyn FnMut(&mut Vec::<(vec4, vec4)>) -> JoinHandle<Vec<Particle>> = &mut |p_to_test| {

            let region = self.region.clone();

            let mut particles = Vec::with_capacity(JOB_SIZE);
            ::std::mem::swap(p_to_test, &mut particles);

            spawn(
                move || {
                    let mut p_to_add = Vec::with_capacity(particles.len());

                    for p in particles {
                        if region.contains(p.0) {
                            let mut particle = Particle::with_pos(p.0);
                            particle.vel = p.1;
                            particle.den = start_density;
                            particle.mat = mat_id;
                            p_to_add.push(particle);
                        }
                    }

                    return p_to_add;
                }
            )
        };


        //loop over every spot in the bound that could house a particle
        let mut offset2 = 0.0;
        loop {
            let mut offset = 0.0;
            pos[1] = bound.min[1]+offset2;
            loop {
                pos[0] = bound.min[0]+(offset+offset2);
                loop {

                    if particles_to_test.len() >= JOB_SIZE {
                        threads.push(dispatch(&mut particles_to_test));
                    }

                    num_in_box += 1;
                    particles_to_test.push((pos.into(), (self.vel)(pos.into())));

                    pos[0] += h;
                    if pos[0] - bound.min[0] > bound.dim[0] || bound.dim[0]==0.0  { break };
                }
                pos[1] += h/2f32.sqrt();
                offset = if offset==0.0 {h/2.0} else {0.0};
                if pos[1] - bound.min[1] > bound.dim[1] || bound.dim[1]==0.0 { break };
            }
            pos[2] += h/3.0f32.sqrt();
            offset2 = if offset2==0.0 {h/8f32.sqrt()} else {0.0};
            if pos[2] - bound.min[2] > bound.dim[2] || bound.dim[2]==0.0 { break };
        }

        if particles_to_test.len() > 0 { threads.push(dispatch(&mut particles_to_test)); }

        //join all of the threads into one list of particles
        let list_of_lists = threads.into_iter().map(|t| t.join().unwrap()).collect::<Vec<_>>();
        let mut list = list_of_lists.into_iter().flatten().collect::<Vec<_>>();

        list.shrink_to_fit();

        //compute the mass of the particles

        let box_mass = start_density as f64 * (0..4).fold(1.0,
            |content, i| (if bound.dim[i]>0.0 {content * bound.dim[i] as f64} else {content})
        );

        let mut mat:Material = self.mat.into();
        mat.mass = (box_mass / (num_in_box as f64)) as f32;

        if unsafe {crate::LOGGING} { println!("{:?}", mat); }

        return (list, mat);

    }
}
