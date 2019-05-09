
use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash,Hasher};

use maths_traits::algebra::*;
use maths_traits::analysis::metric::*;
use free_algebra::*;

use crate::soft_body::*;
use self::shaders::*;

mod shaders;

glsl!{$
    pub use self::types::*;
    mod types {
        @Lib

            public struct Particle {
                float den;
                uint mat;
                vec4 ref_pos;
                vec4 pos;
                vec4 vel;
                mat4 stress;
            };

        @Rust

            impl Particle {

                pub fn with_pos(pos: vec4) -> Self {
                    Self::with_pos_vel(pos, [0.0,0.0,0.0,0.0].into())
                }

                pub fn with_pos_vel(pos: vec4, vel: vec4) -> Self {
                    let k = 0.0;
                    Particle {
                        den: 0.0,
                        mat: 0,
                        ref_pos: pos,
                        pos: pos,
                        vel: vel,
                        stress: [[0.0,0.0,0.0,0.0],[0.0,k,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]].into()
                    }
                }
            }
    }

}

pub type ParticleBuffer = Buffer<[Particle], Read>;

#[derive(Clone)]
pub struct Particles {
    pub buf: ParticleBuffer,
    pub boundary: Rc<ParticleBuffer>
}

impl Particles {
    pub fn new(gl: &GLProvider, particles: Box<[Particle]>, boundary: Box<[Particle]>) -> Self {
        Particles{
            buf: Buffer::readonly_from(gl, particles),
            boundary: Rc::new(Buffer::readonly_from(gl, boundary))
        }
    }

    pub unsafe fn mirror(&self) -> Self {
        Particles {
            buf: Buffer::<[_],_>::uninitialized(&self.buf.gl_provider(), self.buf.len()),
            boundary: self.boundary.clone()
        }
    }
}


#[derive(Clone)]
struct Term(Rc<Particles>);
impl Eq for Term {}
impl PartialEq for Term { #[inline] fn eq(&self, rhs:&Self) -> bool {Rc::ptr_eq(&self.0, &rhs.0)} }
impl Hash for Term {
    #[inline] fn hash<H:Hasher>(&self, h: &mut H) {h.write_usize(self.0.as_ref() as *const _ as usize)}
}

pub struct ParticleState {
    terms: RefCell<FreeModule<GLfloat, Term>>,
    arith: Option<ArithShaders>
}

impl ParticleState {

    pub fn from_list(gl: &GLProvider, particles: Box<[Particle]>, boundary: Box<[Particle]>) -> Self {
        Self::new(Particles::new(gl, particles, boundary))
    }

    pub fn new(particles: Particles) -> Self {
        ParticleState {
            arith: Some(ArithShaders::new(&particles.buf.gl_provider()).unwrap()),
            terms: RefCell::new(Term(Rc::new(particles)).into()),
        }
    }

    fn with_arith(particles: Particles, arith: ArithShaders) -> Self {
        ParticleState {
            terms: RefCell::new(Term(Rc::new(particles)).into()),
            arith: Some(arith)
        }
    }

    fn reduce_with_ref(&self) {
        if self.arith.is_none() {return;}

        let arith = self.arith.as_ref().unwrap();
        let mut lc = self.terms.borrow_mut();

        // print!("{}: ", lc.terms());
        let arr = lc.as_ref().iter().map(
            |(p,r)| (*r,&p.0.buf)
        ).collect::<Vec<_>>();
        // println!();

        if arr.len()==1 && arr[0].0==1.0 {return};

        let prof = unsafe { crate::PROFILER.as_mut().unwrap() };
        prof.new_segment("Arith".to_string());

        let buf = arith.reduce_ref(arr);

        prof.end_segment();

        *lc = match buf {
            Some(b) => Term(Rc::new(Particles{
                buf:b,
                boundary: (lc.as_ref().iter().next().unwrap().0).0.boundary.clone()
            })).into(),
            None => FreeModule::zero(),
        }

    }

    fn reduce(&self) {
        self.reduce_with_ref()

        // if self.arith.is_none() {return;}
        //
        // let arith = self.arith.as_ref().unwrap();
        //
        // let mut lc = self.terms.borrow_mut();
        // let mut temp = Default::default();
        // ::std::mem::swap(&mut temp, &mut *lc);
        //
        // let mut rcs = Vec::with_capacity(temp.terms());
        // let mut ownd = Vec::with_capacity(temp.terms());
        //
        // temp.into_iter().map(|(t,r)| (Rc::try_unwrap(t.0), r)).for_each(
        //     |(res, r)| match res {
        //         Ok(p) => ownd.push((r, p.buf)),
        //         Err(rc) => rcs.push((r, rc)),
        //     }
        // );
        //
        // let refs = rcs.iter().map(|(r,c)| (*r,&c.buf)).collect::<Vec<_>>();
        // if ownd.len()==0 && refs.len()==1 && refs[0].0==1.0 {
        //     *lc = Term(rcs.pop().unwrap().1).into();
        //     return;
        // }
        //
        // let prof = unsafe { crate::PROFILER.as_mut().unwrap() };
        // prof.new_segment("Arith".to_string());
        //
        // let buf = arith.reduce(ownd, refs);
        //
        // prof.end_segment();
        //
        // *lc = match buf {
        //     Some(b) => Term(Rc::new(Particles{ buf:b })).into(),
        //     None => FreeModule::zero(),
        // }
    }

    pub fn map<F:FnOnce(Particles)->Particles>(mut self, f:F) -> Self {
        self.reduce();
        let arith = self.arith.take();
        let particles = {
            self.terms.into_inner().into_iter().next()
                .map(|(mut t,_)| {Rc::make_mut(&mut t.0); Rc::try_unwrap(t.0).unwrap_or_else(|_| panic!())} )
                .map(f)
        };
        ParticleState {
            terms: RefCell::new(match particles {
                Some(p) => Term(Rc::new(p)).into(),
                None => Zero::zero()
            }),
            arith: arith
        }
    }

    pub fn map_into<R,F:FnOnce(Particles)->R>(self, f:F) -> Option<R> {self.map_into_or(None, |p| Some(f(p)))}
    pub fn map_into_or<R,F:FnOnce(Particles)->R>(self, def:R, f:F) -> R {self.map_into_or_else(|| def, f)}
    pub fn map_into_or_else<R,F:FnOnce(Particles)->R,G:FnOnce()->R>(self, def:G, f:F) -> R {
        self.reduce();
        self.terms.into_inner().into_iter().next()
            .map(|(mut t,_)| {Rc::make_mut(&mut t.0); Rc::try_unwrap(t.0).unwrap_or_else(|_| panic!())} )
            .map_or_else(def, f)
    }

    pub fn map_mut<R,F:FnOnce(&mut Particles)->R>(&mut self, f:F) -> Option<R>{ self.map_mut_or(None, |p| Some(f(p))) }
    pub fn map_mut_or<R,F:FnOnce(&mut Particles)->R>(&mut self, def:R, f:F) -> R { self.map_mut_or_else(|| def, f) }
    pub fn map_mut_or_else<R,F:FnOnce(&mut Particles)->R,G:FnOnce()->R>(&mut self, def:G, f:F) -> R {
        self.reduce();
        let mut temp = FreeModule::zero();
        ::std::mem::swap(&mut *self.terms.borrow_mut(), &mut temp);
        let particles = temp
            .into_iter().next()
            .map(|(mut t,_)| {Rc::make_mut(&mut t.0); Rc::try_unwrap(t.0).unwrap_or_else(|_| panic!())} );

        match particles {
            Some(mut p) => {
                let res = f(&mut p);
                *self.terms.borrow_mut() = Term(Rc::new(p)).into();
                res
            },
            None => def()
        }
    }

    pub fn map_ref<R,F:FnOnce(&Particles)->R>(&self, f:F) -> Option<R>{ self.map_ref_or(None, |p| Some(f(p))) }
    pub fn map_ref_or<R,F:FnOnce(&Particles)->R>(&self, def:R, f:F) -> R { self.map_ref_or_else(|| def, f) }
    pub fn map_ref_or_else<R,F:FnOnce(&Particles)->R,G:FnOnce()->R>(&self, def:G, f:F) -> R {
        self.reduce();
        self.terms.borrow().as_ref().iter().next().map(|(t,_)|t.0.as_ref()).map_or_else(def, f)
    }

    pub fn replace(&mut self, p:Particles) { *self.terms.borrow_mut() = Term(Rc::new(p)).into(); }

    pub fn velocity(self) -> Self {

        if self.terms.borrow().terms()==0 || self.arith.is_none() {
            self
        } else if self.terms.borrow().terms()==1 && self.arith.is_some() {
            let arith = self.arith.unwrap();
            let particles = self.terms.into_inner().into_iter().map(
                |(p,r)| {
                    let bdry = p.0.boundary.clone();
                    Particles{
                        buf: match Rc::try_unwrap(p.0) {
                            Ok(particles) => (arith.vel_mut)(r, particles.buf),
                            Err(new_p) => (arith.vel)(&(r, &new_p.buf))
                        },
                        boundary: bdry
                    }
                }
            ).next().unwrap();

            Self::with_arith(particles, arith)
        } else {
            self.reduce();
            self.velocity()
        }
    }


}

impl Clone for ParticleState {
    fn clone(&self) -> Self {
        self.reduce();
        ParticleState {terms: self.terms.clone(), arith: self.arith.clone()}
    }
}

impl AddAssign for ParticleState {
    #[inline] fn add_assign(&mut self, rhs:Self) {
        self.arith = self.arith.take().or(rhs.arith);
        *self.terms.borrow_mut() += rhs.terms.into_inner();
    }
}

impl SubAssign for ParticleState {
    #[inline] fn sub_assign(&mut self, rhs:Self) {
        self.arith = self.arith.take().or(rhs.arith);
        *self.terms.borrow_mut() -= rhs.terms.into_inner();
    }
}

impl Neg for ParticleState {
    type Output = Self;
    #[inline] fn neg(self) -> Self {
        ParticleState {terms: RefCell::new(-self.terms.into_inner()), arith: self.arith.clone()}
    }
}

impl Zero for ParticleState {
    #[inline] fn zero() -> Self { ParticleState {terms: RefCell::new(Zero::zero()), arith: None} }
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

impl InnerProductSpace<GLfloat> for ParticleState {
    fn inner_product(self, rhs: Self) -> GLfloat {self.dot(rhs)}
    fn norm_sqrd(self) -> GLfloat {self.q_form()}
}

impl QuadradicForm<GLfloat> for ParticleState { fn q_form(self) -> GLfloat {self.clone().dot(self)} }
impl BilinearForm<GLfloat> for ParticleState {
    fn dot(self, rhs: Self) -> GLfloat {
        let a = self.arith.clone().or_else(|| rhs.arith.clone());
        if let Some(arith) = a {
            let prof = unsafe { crate::PROFILER.as_mut().unwrap() };
            prof.new_segment("Norm".to_string());
            let dot = self.map_into_or(0.0, |p1| rhs.map_into_or(0.0, |p2| arith.dot(&p1.buf, &p2.buf)));
            prof.end_segment();
            dot
        } else {
            0.0
        }
    }
}
