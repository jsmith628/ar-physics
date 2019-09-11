
use super::*;

glsl!{$

    pub use self::glsl::*;

    mod glsl {
        @Lib

            public struct Material {
                //universal properties
                bool immobile;
                float mass, start_den;

                //fluid properties
                uint state_eq;
                float sound_speed, target_den, visc;

                //elastic properties
                int strain_order;
                float normal_stiffness, shear_stiffness, normal_damp, shear_damp;

                //plastic properties
                bool plastic;
                float yield_strength, work_hardening, work_hardening_exp, kinematic_hardening, thermal_softening;
                float relaxation_time;
            }

        @Rust

            impl Material {
                pub fn is_solid(&self) -> bool { self.normal_stiffness!=0.0 || self.shear_stiffness!=0.0 }
            }

    }
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

#[non_exhaustive]
pub enum StateEquation {
    Zero = 0,
    Constant = 1,
    IdealGas = 2,
    Tait = 3
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

    pub fn new_elastic_solid(den: f32, strain_order: GLint, stiffness: MatTensor, dampening: MatTensor) {
        Self::new_solid(den, strain_order, stiffness, dampening, None);
    }

    pub fn new_plastic_solid(den: f32, strain_order: GLint, stiffness: MatTensor, dampening: MatTensor, strength: PlasticParams) {
        Self::new_solid(den, strain_order, stiffness, dampening, Some(strength));
    }

    pub fn new_solid(den: f32, strain_order: GLint, stiffness: MatTensor, dampening: MatTensor, strength: Option<PlasticParams>) -> Self {
        let mut mat = Material::default();
        mat.start_den = den;
        mat.strain_order = strain_order;
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
