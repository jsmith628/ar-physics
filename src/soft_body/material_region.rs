
use super::*;
use std::rc::Rc;

glsl!{$

    pub use self::glsl::*;

    mod glsl {
        @Lib

            public struct AABB {
                vec4 min;
                vec4 dim;
            };

            public struct AABE {
                vec4 center;
                vec4 radii;
            }

            public struct Material {
                //universal properties
                bool immobile;
                float mass, start_den;

                //fluid properties
                uint state_eq;
                float sound_speed, target_den, visc;

                //elastic properties
                float normal_stiffness, shear_stiffness, normal_damp, shear_damp;

                //plastic properties
                bool plastic;
                float yield_strength, work_hardening, work_hardening_exp, kinematic_hardening, thermal_softening;
                float relaxation_time;
            }

        @Rust

            use super::*;

            impl AABB {

                pub fn center(&self) -> vec4 {
                    [self.min[0]+0.5*self.dim[0],self.min[1]+0.5*self.dim[1],self.min[2]+0.5*self.dim[2],self.min[3]+0.5*self.dim[3]].into()
                }

                pub fn border(&self, b: f32) -> Difference<AABB, AABB> {
                    let (m, c, d) = (self.min, self.center(), self.dim);
                    Difference(
                        *self,
                        AABB {
                            min: [(m[0]+b).min(c[0]), (m[1]+b).min(c[1]), (m[2]+b).min(c[2]), (m[3]+b).min(c[3])].into(),
                            dim: [(d[0]-2.0*b).max(0.0), (d[1]-2.0*b).max(0.0), (d[2]-2.0*b).max(0.0), (d[3]-2.0*b).max(0.0)].into()
                        }
                    )
                }

                pub fn max(&self) -> vec4 {
                    [self.min[0]+self.dim[0],self.min[1]+self.dim[1],self.min[2]+self.dim[2],self.min[3]+self.dim[3]].into()
                }
            }

            impl AABE {
                pub fn border(&self, b: f32) -> Difference<AABE, AABE> {
                    let r = self.radii;
                    Difference(
                        *self,
                        AABE {
                            center: self.center,
                            radii: [(r[0]-b).max(0.0), (r[1]-b).max(0.0), (r[2]-b).max(0.0), (r[3]-b).max(0.0)].into()
                        }
                    )
                }
            }

            impl Region for AABB {
                #[inline] fn bound(&self) -> AABB { *self }
                fn contains(&self, p: vec4) -> bool {
                    for i in 0..4 {
                        if p[i] < self.min[i] || p[i]-self.min[i] > self.dim[i] {
                            return false;
                        }
                    }
                    return true;
                }
            }

            impl Region for AABE {
                #[inline]
                fn bound(&self) -> AABB {
                    let (c, a) = (self.center, self.radii);
                    AABB {
                        min: [c[0]-a[0], c[1]-a[1], c[2]-a[2], c[3]-a[3]].into(),
                        dim: [2.0*a[0], 2.0*a[1], 2.0*a[2], 2.0*a[3]].into()
                    }
                }

                fn contains(&self, p: vec4) -> bool {
                    let lhs = (0..4).fold(0.0, |c, i| {
                        let d = (p[i] - self.center[i]) as f64;
                        let r = self.radii[i] as f64;
                        if r > 0.0 {
                            let coeff = (0..4).fold(1.0, |prod, j| {
                                if self.radii[j]>0.0 && j!=i {
                                    prod * self.radii[i] as f64
                                } else {
                                    prod
                                }
                            });
                            c + d*d*coeff*coeff
                        } else {
                            c
                        }
                    });
                    let rhs = (0..4).fold(1.0, |rhs, i| if self.radii[i]>0.0 {rhs * self.radii[i] as f64} else {rhs});
                    lhs <= rhs * rhs
                }
            }

    }

}

pub trait Region: 'static {
    fn bound(&self) -> AABB;
    fn contains(&self, p: vec4) -> bool;
}

impl<R:Region+?Sized,Ptr: ::std::ops::Deref<Target=R> + 'static> Region for Ptr {
    fn bound(&self) -> AABB {self.deref().bound()}
    fn contains(&self, p: vec4) -> bool { self.deref().contains(p) }
}

#[derive(Clone,PartialEq,Eq)]
pub struct Difference<L:Region, R:Region>(pub L, pub R);
impl<L:Region, R:Region> Region for Difference<L, R> {
    fn bound(&self) -> AABB{self.0.bound()}
    fn contains(&self, p: vec4) -> bool {self.0.contains(p) && !self.1.contains(p)}
}

#[derive(Clone,PartialEq,Eq)]
pub struct Union<L:Region, R:Region>(pub L, pub R);
impl<L:Region, R:Region> Region for Union<L, R> {
    fn bound(&self) -> AABB{
        let (l,r) = (self.0.bound(), self.1.bound());
        let (m1, m2) = (l.max(), r.max());
        let min = [l.min[0].min(r.min[0]),l.min[1].min(r.min[1]),l.min[2].min(r.min[2]),l.min[3].min(r.min[3])];
        let max = [m1[0].max(m2[0]), m1[1].max(m2[1]), m1[2].max(m2[2]), m1[3].max(m2[3])];
        AABB{
            min: min.into(),
            dim: [max[0]-min[0],max[1]-min[1],max[2]-min[2],max[3]-min[3]].into(),
        }
    }
    fn contains(&self, p: vec4) -> bool {self.0.contains(p) || self.1.contains(p)}
}

#[derive(Clone,PartialEq,Eq)]
pub struct Intersection<L:Region, R:Region>(pub L, pub R);
impl<L:Region, R:Region> Region for Intersection<L, R> {
    fn bound(&self) -> AABB{
        let (l,r) = (self.0.bound(), self.1.bound());
        let (max1, max2) = (l.max(), r.max());

        let mut intersection = AABB{
            min: [0.0,0.0,0.0,0.0].into(),
            dim: [0.0,0.0,0.0,0.0].into(),
        };

        for i in 0..4 {
            intersection.min[i] = l.min[i].max(r.min[i]);
            intersection.dim[i] = max1[i].min(max2[i]) - intersection.min[i];
        }

        intersection

    }
    fn contains(&self, p: vec4) -> bool {self.0.contains(p) && self.1.contains(p)}
}

pub struct MatTensor {
    pub normal: f32,
    pub shear: f32
}

pub struct PlasticParams {
    pub yield_strength: f32,
    pub work_hardening: f32,
    pub work_hardening_exp: f32,
    pub kinematic_hardening: f32,
    pub thermal_softening: f32,
    pub relaxation_time: f32
}

pub enum StateEquation {
    Zero = 0,
    Tait = 1,
    IdealGas = 2
}

impl Material {

    pub fn new_liquid(den: f32, sound_speed: f32, visc: f32) -> Self {
        Self::new_fluid(den, den, sound_speed, visc, StateEquation::Tait)
    }

    pub fn new_gas(start_den: f32, target_den: f32, sound_speed: f32, visc: f32) -> Self {
        Self::new_fluid(start_den, target_den, sound_speed, visc, StateEquation::IdealGas)
    }

    pub fn new_fluid(start_den: f32, target_den: f32, sound_speed: f32, visc: f32, state: StateEquation) -> Self {
        let mut mat = Material::default();
        mat.start_den = start_den;
        mat.target_den = target_den;
        mat.sound_speed = sound_speed;
        mat.visc = visc;
        mat.state_eq = state as GLuint;
        mat
    }

    pub fn new_elastic_solid(den: f32, stiffness: MatTensor, dampening: MatTensor) {
        Self::new_solid(den, stiffness, dampening, None);
    }

    pub fn new_plastic_solid(den: f32, stiffness: MatTensor, dampening: MatTensor, strength: PlasticParams) {
        Self::new_solid(den, stiffness, dampening, Some(strength));
    }

    pub fn new_solid(den: f32, stiffness: MatTensor, dampening: MatTensor, strength: Option<PlasticParams>) -> Self {
        let mut mat = Material::default();
        mat.start_den = den;
        mat.normal_stiffness = stiffness.normal;
        mat.shear_stiffness = stiffness.shear;
        mat.normal_damp = dampening.normal;
        mat.shear_damp = dampening.shear;

        if let Some(params) = strength {
            mat.plastic = true.into();
            mat.yield_strength = params.yield_strength;
            mat.work_hardening = params.work_hardening;
            mat.work_hardening_exp = params.work_hardening_exp;
            mat.kinematic_hardening = params.kinematic_hardening;
            mat.thermal_softening = params.thermal_softening;
            mat.relaxation_time = params.relaxation_time;
        }

        mat

    }

    pub fn new_immobile(friction: f32) -> Self {
        let mut mat = Material::default();
        mat.visc = friction;
        mat.start_den = 1.0;
        mat.target_den = 1.0;
        mat.immobile = true.into();
        mat
    }

}

pub type Materials = Buffer<[Material], CopyOnly>;

#[derive(Clone)]
pub struct MaterialRegion {
    pub region: Rc<Region>,
    pub packing_coefficient: f32,
    pub mat: Material
}

impl MaterialRegion {

    pub fn new<R:Region>(region: R, packing: f32, mat: Material) -> Self {
        MaterialRegion {
            region: Rc::new(region),
            packing_coefficient: packing,
            mat: mat
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
