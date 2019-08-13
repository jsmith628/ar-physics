
use super::*;

glsl!{$

    pub use self::glsl::*;

    mod glsl {
        @Lib
            public struct MatInteraction {
                uint potential;
                float strength, radius;
                float dampening, friction;
            }

    }

}

#[non_exhaustive]
pub enum ContactPotental {
    Zero = 0,
    Constant = 1,
    Linear = 2,
    Tait = 3,
    LennardJones = 4,
    Hertz = 5
}

impl MatInteraction {

    pub fn new(ty: ContactPotental, strength: GLfloat, radius: GLfloat, damp: GLfloat, friction: GLfloat) -> Self {
        MatInteraction {
            potential: ty as GLuint,
            strength: strength,
            radius: radius,
            dampening: damp,
            friction: friction
        }
    }

    pub fn default_between(mat1:Material, mat2:Material, h:GLfloat) -> Self {
        if mat1.state_eq != StateEquation::Zero as GLuint && mat2.state_eq != StateEquation::Zero as GLuint {
            Self::default()
        } else {
            MatInteraction::new(ContactPotental::Linear, 10000.0, h*2.0, 0.0, mat1.visc+mat2.visc)
            // MatInteraction::new(ContactPotental::Tait, 0.05, 2.0*h, 0.0, 0.0)
            // MatInteraction::new(ContactPotental::LennardJones, 0.01, h, 0.0, 0.0)
        }
    }

}
