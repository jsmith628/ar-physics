
use super::*;
use std::rc::Rc;

pub use self::region::*;
pub use self::material::*;

mod region;
mod material;

#[derive(Clone)]
pub struct MaterialRegion {
    pub region: Rc<dyn Region>,
    pub packing_coefficient: f32,
    pub mat: Material,
    pub vel: Rc<dyn Fn(vec4)->vec4>
}

impl MaterialRegion {

    pub fn new<R:Region>(region: R, packing: f32, mat: Material) -> Self {
        MaterialRegion {
            region: Rc::new(region),
            packing_coefficient: packing,
            mat: mat,
            vel: Rc::new(|_| vec4::default())
        }
    }

    pub fn with_vel<R:Region, V:Fn(vec4)->vec4+'static>(region: R, packing: f32, mat: Material, v:V) -> Self {
        MaterialRegion {
            region: Rc::new(region),
            packing_coefficient: packing,
            mat: mat,
            vel: Rc::new(v)
        }
    }

    pub fn gen_particles(&self, mut h: f32, mat_id: u32) -> (Vec<Particle>, Material) {

        h = self.packing_coefficient * h;

        let bound = self.region.bound();
        let mut list = Vec::with_capacity((0..4).fold(1, |c, i| c*(1.0f32.max(bound.dim[i]/h) as usize)));

        let mut num_in_box = 0u64;
        let mut pos = bound.min;
        let start_density = self.mat.start_den;

        let mut offset2 = 0.0;
        loop {
            let mut offset = 0.0;
            pos[1] = bound.min[1]+offset2;
            loop {
                pos[0] = bound.min[0]+(offset+offset2);
                loop {

                    num_in_box += 1;
                    if self.region.contains(pos) {
                        let mut particle = Particle::with_pos(pos.into());
                        particle.vel = (self.vel)(particle.pos);
                        particle.den = start_density;
                        particle.mat = mat_id;
                        list.push(particle);
                    }

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
        list.shrink_to_fit();

        let box_mass = start_density as f64 * (0..4).fold(1.0,
            |content, i| (if bound.dim[i]>0.0 {content * bound.dim[i] as f64} else {content})
        );

        let mut mat:Material = self.mat.into();
        mat.mass = (box_mass / (num_in_box as f64)) as f32;

        println!("{:?}", mat);

        return (list, mat);

    }
}
