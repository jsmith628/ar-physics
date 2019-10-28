
use std::rc::*;
use std::cell::RefCell;
use std::hash::{Hash,Hasher};

use maths_traits::algebra::*;
use maths_traits::analysis::metric::{InnerProductSpace};
use free_algebra::*;

use crate::soft_body::*;
use self::particles::*;
use self::buf_vec::*;

pub use particles::{Particles, ParticleBuffer, ParticleVec, SolidParticleBuffer};

mod particles;
pub mod buf_vec;

glsl!{$
    pub use self::types::*;
    mod types {
        @Lib

            public struct Particle {
                float den;
                uint mat, solid_id;
                vec4 pos;
                vec4 vel;
            };

            public struct SolidParticle {
                uint part_id;
                vec4 ref_pos;
                mat4 stress;
            }

        @Rust

            impl Particle {

                pub fn with_pos(pos: vec4) -> Self {
                    Self::with_pos_vel(pos, [0.0,0.0,0.0,0.0].into())
                }

                pub fn with_pos_vel(pos: vec4, vel: vec4) -> Self {
                    Particle {
                        den: 0.0,
                        mat: 0, solid_id: !0,
                        pos: pos,
                        vel: vel,
                    }
                }
            }

            impl SolidParticle {

                pub fn new(id:GLuint, pos: vec4) -> Self {
                    SolidParticle {
                        part_id: id,
                        ref_pos: pos,
                        stress: [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]].into()
                    }
                }

            }
    }

}

#[derive(Clone)]
struct Term(Rc<Particles>, ArithShaders);
impl Eq for Term {}
impl PartialEq for Term { #[inline] fn eq(&self, rhs:&Self) -> bool {Rc::ptr_eq(&self.0, &rhs.0)} }
impl Hash for Term {
    #[inline] fn hash<H:Hasher>(&self, h: &mut H) {h.write_usize(self.0.as_ref() as *const _ as usize)}
}

pub struct ParticleState {
    terms: RefCell<FreeModule<GLfloat, Term>>
}

impl ParticleState {

    pub fn new(particles: Particles) -> Self {
        let arith = ArithShaders::new(&particles.particles().gl_provider()).unwrap();
        Self::with_arith(particles, arith)
    }

    fn with_arith(particles: Particles, arith: ArithShaders) -> Self {
        ParticleState { terms: RefCell::new(Term(Rc::new(particles), arith).into()) }
    }

    fn reduce_with_ref(&self) {
        let prof = unsafe { crate::PROFILER.as_mut().unwrap() };
        prof.new_segment("Arith".to_string());

        let mut lc = self.terms.borrow_mut();
        let buf = {
            if let Some(arith) = lc.iter().next().map(|(_,t)| &t.1) {
                let arr = lc.iter().map(|(r,p)| (*r,&*p.0)).collect::<Vec<_>>();
                arith.linear_combination(arr).map(|b| Term(Rc::new(b),arith.clone()))
            } else {
                None
            }
        };
        *lc = buf.map_or_else(|| Zero::zero(), |t| t.into());

        prof.end_segment();

    }

    fn reduce(&self) {
        self.reduce_with_ref()
    }

    pub fn map<F:FnOnce(Particles)->Particles>(self, f:F) -> Self {
        self.reduce();
        ParticleState {
            terms: RefCell::new(
                self.terms.into_inner().into_iter().next().map_or_else(
                    || Zero::zero(),
                    |(_,mut t)| {
                        Rc::make_mut(&mut t.0);
                        let p = Rc::try_unwrap(t.0).unwrap_or_else(|_| panic!());
                        Term(Rc::new(f(p)), t.1.clone()).into()
                    }
                )
            )
        }
    }

    pub fn map_terms<F:FnOnce(Particles)->Vec<Particles>>(self, f:F) -> Self {
        self.reduce();
        ParticleState {
            terms: RefCell::new(
                self.terms.into_inner().into_iter().next().map_or_else(
                    || Zero::zero(),
                    |(_,Term(mut p, arith))| {
                        Rc::make_mut(&mut p);
                        let p = Rc::try_unwrap(p).unwrap_or_else(|_| panic!());
                        f(p).into_iter().map(|p2| Term(Rc::new(p2), arith.clone())).sum()
                    }
                )
            )
        }
    }

    pub fn map_into<R,F:FnOnce(Particles)->R>(self, f:F) -> Option<R> {self.map_into_or(None, |p| Some(f(p)))}
    pub fn map_into_or<R,F:FnOnce(Particles)->R>(self, def:R, f:F) -> R {self.map_into_or_else(|| def, f)}
    pub fn map_into_or_else<R,F:FnOnce(Particles)->R,G:FnOnce()->R>(self, def:G, f:F) -> R {
        self.reduce();
        self.terms.into_inner().into_iter().next().map(
            |(_,mut t)| {
                Rc::make_mut(&mut t.0);
                Rc::try_unwrap(t.0).unwrap_or_else(|_| panic!())
            }
        ).map_or_else(def, f)
    }

    pub fn map_mut<R,F:FnOnce(&mut Particles)->R>(&mut self, f:F) -> Option<R>{ self.map_mut_or(None, |p| Some(f(p))) }
    pub fn map_mut_or<R,F:FnOnce(&mut Particles)->R>(&mut self, def:R, f:F) -> R { self.map_mut_or_else(|| def, f) }
    pub fn map_mut_or_else<R,F:FnOnce(&mut Particles)->R,G:FnOnce()->R>(&mut self, def:G, f:F) -> R {
        self.reduce();
        let mut temp = FreeModule::zero();
        ::std::mem::swap(&mut *self.terms.borrow_mut(), &mut temp);

        let mut collection: Vec<_> = temp.into_iter().collect();
        let result = collection.iter_mut().next().map_or_else(def, |t| f(Rc::make_mut(&mut (t.1).0)));
        *self.terms.borrow_mut() = collection.into_iter().sum();

        result
    }

    pub fn map_ref<R,F:FnOnce(&Particles)->R>(&self, f:F) -> Option<R>{ self.map_ref_or(None, |p| Some(f(p))) }
    pub fn map_ref_or<R,F:FnOnce(&Particles)->R>(&self, def:R, f:F) -> R { self.map_ref_or_else(|| def, f) }
    pub fn map_ref_or_else<R,F:FnOnce(&Particles)->R,G:FnOnce()->R>(&self, def:G, f:F) -> R {
        self.reduce();
        self.terms.borrow().iter().next().map(|(_,t)|t.0.as_ref()).map_or_else(def, f)
    }

    pub fn replace(&mut self, p:Particles) {
        let arith = match self.terms.borrow().iter().next() {
            Some((_,t)) => t.1.clone(),
            None => ArithShaders::new(&p.particles().gl_provider()).unwrap()
        };
        *self.terms.borrow_mut() = Term(Rc::new(p), arith).into();
    }

    pub fn velocity(self) -> Self {

        if self.terms.borrow().num_terms()==0 {
            self
        } else if self.terms.borrow().num_terms()==1 {
            let particles = self.terms.into_inner().into_iter().map(
                |(r,Term(p,arith))| Term(Rc::new(arith.velocity((r,&*p))), arith)
            ).next().unwrap();

            Self { terms: RefCell::new(particles.into()) }
        } else {
            self.reduce();
            self.velocity()
        }
    }

}

impl Clone for ParticleState {
    fn clone(&self) -> Self {
        self.reduce();
        ParticleState {terms: self.terms.clone()}
    }
}

impl AddAssign for ParticleState {
    #[inline] fn add_assign(&mut self, rhs:Self) {*self.terms.borrow_mut()+=rhs.terms.into_inner();}
}

impl SubAssign for ParticleState {
    #[inline] fn sub_assign(&mut self, rhs:Self) { *self.terms.borrow_mut()-=rhs.terms.into_inner();}
}

impl Neg for ParticleState {
    type Output = Self;
    #[inline] fn neg(self) -> Self { ParticleState {terms: RefCell::new(-self.terms.into_inner())} }
}

impl Zero for ParticleState {
    #[inline] fn zero() -> Self { ParticleState {terms: RefCell::new(Zero::zero())} }
    #[inline] fn is_zero(&self) -> bool { self.terms.borrow().is_zero() }
}

impl MulAssign<GLfloat> for ParticleState { #[inline] fn mul_assign(&mut self, rhs:GLfloat) {*self.terms.borrow_mut()*=rhs;}}
impl DivAssign<GLfloat> for ParticleState { #[inline] fn div_assign(&mut self, rhs:GLfloat) {*self.terms.borrow_mut()/=rhs;}}

impl Add for ParticleState { type Output=Self; #[inline] fn add(mut self, rhs:Self)->Self {self+=rhs; self} }
impl Sub for ParticleState { type Output=Self; #[inline] fn sub(mut self, rhs:Self)->Self {self-=rhs; self} }
impl Mul<GLfloat> for ParticleState { type Output=Self; #[inline] fn mul(mut self, rhs:GLfloat)->Self {self*=rhs; self} }
impl Div<GLfloat> for ParticleState { type Output=Self; #[inline] fn div(mut self, rhs:GLfloat)->Self {self/=rhs; self} }

impl AddAssociative for ParticleState {}
impl AddCommutative for ParticleState {}
impl Distributive<GLfloat> for ParticleState {}

impl InnerProductSpace<GLfloat> for ParticleState {
    fn inner_product(self, rhs: Self) -> GLfloat {
        let prof = unsafe { crate::PROFILER.as_mut().unwrap() };
        prof.new_segment("Norm".to_string());

        if self.terms.borrow().num_terms()>1 {self.reduce();}
        let dot = self.terms.into_inner().into_iter().next().map_or_else(
            || 0.0,
            |(r1,Term(p1,arith))| {
                if rhs.terms.borrow().num_terms()>1 {rhs.reduce();}
                rhs.terms.into_inner().into_iter().next().map_or_else(
                    || 0.0,
                    |(r2,Term(p2,_))| r1 * r2 * arith.dot(&p1.particles(), &p2.particles())
                )
            }
        );

        prof.end_segment();
        dot
    }
}
