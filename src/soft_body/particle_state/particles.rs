
macro_rules! gen_lin_comb{

    ($d:tt $shdr1:ident $shdr2:ident $(($r:ident $p:ident $p_alt:ident))* ) => {
        gen_lin_comb!($shdr1 $shdr2 {$(($r $p $p_alt))*} {$(($r $p $p_alt))*} $d);
    };

    ($shdr1:ident $shdr2:ident
        {$(($r:ident $p:ident $p_alt:ident))*}
        {($r0:ident $p0:ident $p_alt0:ident) $(($r2:ident $p2:ident $p_alt2:ident))*}
        $d:tt
    ) => {

        glsl!{$d
            mod $shdr1 {
                @Rust
                    use super::*;
                    use std::mem::transmute;
                    use std::cell::RefCell;

                    impl Program {
                        pub(super) fn into_closure(self) -> LCClosure {
                            const N: usize = macro_program!([$($r)*] @count @num_expr @return);
                            let shader = RefCell::new(self);
                            Box::new(
                                move |arr| {
                                    let (split, _) = arr.split_at(N);
                                    let len = split.iter().fold(usize::min_value(), |l,p| l.max(p.1.len())) as GLuint;
                                    let gl = split[0].1.gl_provider();
                                    if let [$(($r, $p)),*] = split {
                                        unsafe {
                                            let mut dest = Buffer::<[_],_>::uninitialized(&gl, len as usize);
                                            let [$($p_alt),*] = transmute::<[&ParticleBuffer;N], [&mut ParticleBuffer;N]>([$($p),*]);

                                            let mut shdr = shader.borrow_mut();
                                            $(*shdr.$r = *$r;)*

                                            shdr.compute(units(len), 1, 1, $($p_alt),*, &mut dest);
                                            return dest;
                                        }
                                    }
                                    panic!("Array split too small")
                                }
                            )
                        }
                    }

                @Compute

                    #version 460
                    const vec4 VZERO = vec4(0,0,0,0);
                    const mat4 MZERO = mat4(VZERO,VZERO,VZERO,VZERO);

                    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

                    extern struct Particle;

                    $(uniform float $r = 0.0;)*
                    $(layout(std430) buffer $p_alt {readonly Particle $p[];};)*
                    layout(std430) buffer dest {writeonly restrict Particle dst[];};

                    void main() {
                        uint id = gl_GlobalInvocationID.x;

                        if(id < $p0.length()) {
                            dst[id].mat = $p0[id].mat;
                        }
                        $(else if(id < $p2.length()) {dst[id].mat = $p2[id].mat;})*

                        dst[id].den = (id<$p0.length() ? $r0*$p0[id].den : 0) $(+(id<$p2.length() ? $r2*$p2[id].den : 0))*;
                        dst[id].ref_pos = (id<$p0.length() ? $r0*$p0[id].ref_pos : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].ref_pos : VZERO))*;
                        dst[id].pos = (id<$p0.length() ? $r0*$p0[id].pos : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].pos : VZERO))*;
                        dst[id].vel = (id<$p0.length() ? $r0*$p0[id].vel : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].vel : VZERO))*;
                        dst[id].stress = (id<$p0.length() ? $r0*$p0[id].stress : MZERO) $(+(id<$p2.length() ? $r2*$p2[id].stress : MZERO))*;

                    }
            }

            mod $shdr2 {
                @Rust
                    use super::*;
                    use std::mem::transmute;
                    use std::cell::RefCell;

                    impl Program {
                        pub(super) fn into_closure(self) -> SumClosure {
                            const N: usize = macro_program!([$($r)*] @count @num_expr @return);
                            let shader = RefCell::new(self);
                            Box::new(
                                move |$r0, mut $p0, arr| {
                                    let (split, _) = arr.split_at(N-1);
                                    let len = $p0.len().min(split.iter().fold(usize::max_value(), |l,p| l.min(p.1.len()))) as GLuint;
                                    if let [$(($r2, $p2)),*] = split {
                                        unsafe {
                                            let [$($p_alt2),*] = transmute::<[&ParticleBuffer;N-1], [&mut ParticleBuffer;N-1]>([$($p2),*]);

                                            let mut shdr = shader.borrow_mut();
                                            *shdr.$r0 = $r0;
                                            $(*shdr.$r2 = *$r2;)*

                                            shdr.compute(units(len), 1, 1, &mut $p0, $($p_alt2),*);
                                            return $p0;
                                        }
                                    }
                                    panic!("Array split too small")
                                }
                            )
                        }
                    }

                @Compute
                    #version 460
                    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

                    extern struct Particle;

                    uniform float $r0 = 0.0;
                    $(uniform float $r2 = 0.0;)*

                    layout(std430) buffer $p_alt0 {restrict Particle $p0[];};
                    $(layout(std430) buffer $p_alt2 {readonly Particle $p2[];};)*

                    void main() {
                        uint id = gl_GlobalInvocationID.x;
                        $p0[id].den = $r0*$p0[id].den $( + $r2*$p2[id].den)*;
                        $p0[id].ref_pos = $r0*$p0[id].ref_pos $( + $r2*$p2[id].ref_pos)*;
                        $p0[id].pos = $r0*$p0[id].pos $( + $r2*$p2[id].pos)*;
                        $p0[id].vel = $r0*$p0[id].vel $( + $r2*$p2[id].vel)*;
                        $p0[id].stress = $r0*$p0[id].stress $( + $r2*$p2[id].stress)*;
                    }
            }
        }

    };
}

glsl!{$

    mod vel {
        @Rust
            use super::*;
            use std::mem::transmute;
            use std::cell::RefCell;

            impl Program {
                pub(super) fn into_closure(self) -> Box<dyn for<'a> Fn(&'a Term<'a>) -> ParticleBuffer> {
                    let shader = RefCell::new(self);
                    Box::new(
                        move |(r, p)| {
                            let mut shdr = shader.borrow_mut();
                            *shdr.r1 = *r;

                            #[allow(mutable_transmutes)]
                            unsafe {
                                let mut dest = Buffer::<[_],_>::uninitialized(&p.gl_provider(), p.len());
                                let p_mut = transmute::<&ParticleBuffer, &mut ParticleBuffer>(p);
                                shdr.compute(units(p.len() as GLuint), 1, 1, p_mut, &mut dest);
                                dest
                            }
                        }
                    )
                }
            }

        @Compute
            #version 460
            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            extern struct Particle;

            uniform float r1 = 1.0;
            layout(std430) buffer part_ {readonly restrict Particle part[];};
            layout(std430) buffer dest_ {writeonly restrict Particle dest[];};

            void main() {
                uint id = gl_GlobalInvocationID.x;
                dest[id].mat = part[id].mat;
                dest[id].den = 0.0;
                dest[id].ref_pos = vec4(0.0,0.0,0.0,0.0);
                dest[id].pos = r1*part[id].vel;
                dest[id].vel = vec4(0.0,0.0,0.0,0.0);
                dest[id].stress = mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0));
            }
    }

    mod vel_mut {
        @Rust
            use super::*;
            use std::cell::RefCell;

            impl Program {
                pub(super) fn into_closure(self) -> Box<dyn Fn(GLfloat, ParticleBuffer) -> ParticleBuffer> {
                    let shader = RefCell::new(self);
                    Box::new(
                        move |r, mut p| {
                            let mut shdr = shader.borrow_mut();
                            *shdr.r1 = r;
                            shdr.compute(units(p.len() as GLuint), 1, 1, &mut p);
                            return p;
                        }
                    )
                }
            }

        @Compute
            #version 460
            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            extern struct Particle;

            uniform float r1 = 1.0;
            layout(std430) buffer part_ {restrict Particle part[];};

            void main() {
                uint id = gl_GlobalInvocationID.x;
                part[id].den = 0.0;
                part[id].ref_pos = vec4(0.0,0.0,0.0,0.0);
                part[id].pos = r1*part[id].vel;
                part[id].vel = vec4(0.0,0.0,0.0,0.0);
                part[id].stress = mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0));
            }
    }

    mod dot {
        @Rust use super::Particle;
        @Compute
            #version 460
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            extern struct Particle;
            layout(std430) buffer part1_ {readonly Particle p1[];};
            layout(std430) buffer part2_ {readonly Particle p2[];};
            layout(std430) buffer norm_list {restrict writeonly float results[];};

            const uint gids = gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z;
            shared float norm[gids];

            void main() {

                uint gid = gl_LocalInvocationIndex;
                uint id = gl_GlobalInvocationID.x;

                //first, compute the norm_squared of each particle
                if(id < p1.length() && id < p2.length()) {
                    norm[gid] =
                        p1[id].den*p2[id].den +
                        dot(p1[id].pos,p2[id].pos) +
                        dot(p1[id].vel,p2[id].vel);
                } else {
                    norm[gid] = 0;
                }
                barrier();

                //then, sum up every value by iterated halfing of local invocations
                uint shadow = gids;
                while(shadow > 1) {
                    //gids is always a power of two, so we can just divide by two with bitshifts
                    shadow = shadow << 2;

                    //add the lower invoc with its upper pair
                    // if(gid < shadow) norm[gid] += norm[gid+shadow];
                    if(gid < shadow)
                        norm[gid] += norm[gid+shadow]+norm[gid+shadow+2]+norm[gid+shadow*3];

                    //make sure every invoc gets to this point
                    barrier();
                }

                //store the work groups results in the appropriate spot in the buffer
                if(gid==0) results[gl_WorkGroupID.x] = norm[0];

            }

    }




}

gen_lin_comb!($ lin_comb_1 sum_1 (r1 p1 _p1) );
gen_lin_comb!($ lin_comb_2 sum_2 (r1 p1 _p1) (r2 p2 _p2) );
gen_lin_comb!($ lin_comb_3 sum_3 (r1 p1 _p1) (r2 p2 _p2) (r3 p3 _p3) );
gen_lin_comb!($ lin_comb_4 sum_4 (r1 p1 _p1) (r2 p2 _p2) (r3 p3 _p3) (r4 p4 _p4) );
gen_lin_comb!($ lin_comb_5 sum_5 (r1 p1 _p1) (r2 p2 _p2) (r3 p3 _p3) (r4 p4 _p4) (r5 p5 _p5) );

use super::*;
use std::rc::Rc;
use maths_traits::analysis::ComplexSubset;

pub(self) use super::Particle;
pub(self) fn units(p: GLuint) -> GLuint { ComplexSubset::ceil((p) as GLfloat / 128.0) as GLuint }

pub(self) type Term<'a> = (GLfloat, &'a ParticleBuffer);
pub(self) type OwnedTerm = (GLfloat, ParticleBuffer);
pub(self) type LCClosure = Box<dyn for<'a> Fn(&'a [Term]) -> ParticleBuffer>;
pub(self) type SumClosure = Box<dyn for<'a> Fn(GLfloat, ParticleBuffer, &'a [Term<'a>]) -> ParticleBuffer>;

#[derive(Clone)]
pub struct ArithShaders {
    lc: Rc<[LCClosure]>,
    sum: Rc<[SumClosure]>,
    vel: Rc<dyn for<'a> Fn(&'a Term<'a>) -> ParticleBuffer>,
    vel_mut: Rc<dyn Fn(GLfloat, ParticleBuffer) -> ParticleBuffer>,
    dot: Rc<dot::Program>,
}

impl ArithShaders {

    pub fn new(gl: &GLProvider) -> Result<Self, GLError> {
        Ok(ArithShaders {
            lc: Rc::new(
                [
                    lin_comb_1::init(gl)?.into_closure(),
                    lin_comb_2::init(gl)?.into_closure(),
                    lin_comb_3::init(gl)?.into_closure(),
                    lin_comb_4::init(gl)?.into_closure(),
                    lin_comb_5::init(gl)?.into_closure(),
                ]
            ),
            sum: Rc::new(
                [
                    sum_1::init(gl)?.into_closure(),
                    sum_2::init(gl)?.into_closure(),
                    sum_3::init(gl)?.into_closure(),
                    sum_4::init(gl)?.into_closure(),
                    sum_5::init(gl)?.into_closure(),
                ]
            ),
            vel: vel::init(gl)?.into_closure().into(),
            vel_mut: vel_mut::init(gl)?.into_closure().into(),
            dot: Rc::new(dot::init(gl)?)
        })
    }

    pub fn dot(&self, buf1: &ParticleBuffer, buf2: &ParticleBuffer) -> GLfloat {
        #[allow(mutable_transmutes)]
        unsafe {
            let len = buf1.len().min(buf2.len());
            let mut results = Buffer::<[_],Read>::uninitialized(&buf1.gl_provider(), ComplexSubset::ceil(len as GLfloat / 64.0) as usize);
            let ub1 = ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(buf1);
            let ub2 = ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(buf2);

            self.dot.compute(len as GLuint,1,1, ub1, ub2, &mut results);

            let map = results.map();
            map.into_iter().sum()
        }
    }

    #[allow(dead_code)]
    fn reduce<'a>(&self, mut owned: Vec<OwnedTerm>, borrowed: Vec<Term<'a>>) -> Option<ParticleBuffer> {
        if owned.len()==0 { return self.reduce_ref(borrowed); }
        if borrowed.len()==0 && owned.len()==1 && owned[0].0==1.0 {
            return Some(owned.pop().unwrap().1);
        }

        let cap = (borrowed.len() + owned.len()) / self.lc.len() + 1;

        let mut owned = owned.into_iter();
        let mut borrowed = borrowed.into_iter();

        let mut new_owned = Vec::with_capacity(cap);
        let mut new_borrowed = Vec::with_capacity(cap);

        loop {

            if let Some((r0,p0)) = owned.next() {
                let mut refs = Vec::with_capacity(self.sum.len()-1);
                let mut ownd = Vec::new();

                while refs.len() < self.sum.len()-1 {
                    match borrowed.next() {
                        Some((r,p)) => refs.push((r,p)),
                        None => break
                    }
                }

                while refs.len() < self.sum.len()-1 {
                    match owned.next() {
                        Some((r,p)) => ownd.push((r,p)),
                        None => break
                    }
                }
                ownd.iter().for_each(|(r,p)| refs.push((*r,p)));

                new_owned.push(
                    (1.0, (self.sum[refs.len()])(
                        r0,p0,refs.as_slice()
                    ))
                );
            } else {
                borrowed.for_each(|t| new_borrowed.push(t));
                break;
            }

        }

        self.reduce(new_owned, new_borrowed)

    }

    fn reduce_owned(&self, mut terms: Vec<OwnedTerm>) -> Option<ParticleBuffer> {
        // self.reduce(terms, Vec::new())
        if terms.len() == 0 { return None; }
        if terms.len() == 1 && terms[0].0 == 1.0 { return Some(terms.pop().unwrap().1); }
        self.reduce_owned(
            terms.chunks(self.lc.len()).map(
                |arr|
                (1.0, (self.lc[arr.len()-1])(
                    arr.iter().map(|(r, b)| (*r, b)).collect::<Vec<_>>().as_slice()
                ))
            ).collect()
        )
    }

    fn reduce_ref<'a>(&self, terms: Vec<Term<'a>>) -> Option<ParticleBuffer> {
        if terms.len() == 0 { return None; }
        self.reduce_owned(
            terms.chunks(self.lc.len()).map(
                |arr| (1.0, (self.lc[arr.len()-1])(arr))
            ).collect()
        )
    }

    pub fn linear_combination<'a>(&self, terms: Vec<(GLfloat, &'a Particles)>) -> Option<Particles> {
        self.reduce_ref(terms.iter().map(|(r,p)| (*r,p.particles())).collect()).map(
            |p| {

                let latest = terms.iter().fold(terms[0].1,
                    |m,p| if p.1.time_id > m.time_id {p.1} else {m}
                );

                Particles {
                    buf: p,
                    solids: latest.solids.clone(),
                    boundary: latest.boundary.clone(),
                    materials: latest.materials.clone(),
                    time_id: latest.time_id
                }
            }
        )
    }

    pub fn velocity<'a>(&self, term: (GLfloat, &'a Particles)) -> Particles {
        Particles {
            buf: (self.vel)(&(term.0, term.1.particles())),
            solids: term.1.solids.clone(),
            boundary: term.1.boundary.clone(),
            materials: term.1.materials.clone(),
            time_id: term.1.time_id
        }
    }

}


pub type ParticleBuffer = Buffer<[Particle], ReadWrite>;
pub type SolidParticleBuffer = Buffer<[SolidParticle], ReadWrite>;
pub type ParticleVec = BufVec<[Particle]>;

#[derive(Clone)]
pub struct Particles {
    buf: ParticleBuffer,
    solids: SolidParticleBuffer,
    boundary: Rc<ParticleBuffer>,
    materials: Rc<Materials>,
    time_id: GLuint
}

unsafe fn add_to_buffer<T:Copy>(old: &Buffer<[T],ReadWrite>, extra: Box<[T]>) -> Buffer<[T],ReadWrite> {
    let mut new_buf = Buffer::<[_],_>::uninitialized(&old.gl_provider(), old.len() + extra.len());

    gl::BindBuffer(gl::COPY_READ_BUFFER, old.id());
    gl::BindBuffer(gl::COPY_WRITE_BUFFER, new_buf.id());

    gl::CopyBufferSubData(
        gl::COPY_READ_BUFFER, gl::COPY_WRITE_BUFFER,
        0, 0, old.data_size() as GLsizeiptr
    );

    gl::BindBuffer(gl::COPY_READ_BUFFER, 0);
    gl::BindBuffer(gl::COPY_WRITE_BUFFER, 0);

    let len = new_buf.len();
    new_buf.map_mut()[old.len()..len].copy_from_slice(&extra);

    new_buf
}

impl Particles {

    fn init_solid_particles(particles: &mut [Particle], materials: &[Material], offset:usize, p_offset:usize) -> Box<[SolidParticle]> {
        let mut result = Vec::new();
        let mut i = p_offset;

        for p in particles {
            if materials[p.mat as usize].normal_stiffness!=0.0 || materials[p.mat as usize].shear_stiffness!=0.0 {
                p.solid_id = (result.len()+offset) as GLuint;
                result.push(SolidParticle::new((i+p_offset)as GLuint,p.pos));
                i += 1;
            }
        }

        if result.len() == 0 { result.push(SolidParticle::new(!0,[0.0,0.0,0.0,0.0].into())) }

        result.into_boxed_slice()
    }

    pub fn new(gl: &GLProvider, materials: Box<[Material]>, mut particles: Box<[Particle]>, boundary: Box<[Particle]>) -> Self {
        Particles{
            solids: Buffer::from_box(gl, Self::init_solid_particles(&mut particles, &materials, 0,0)),
            buf: Buffer::from_box(gl, particles),
            boundary: Rc::new(Buffer::from_box(gl, boundary)),
            materials: Rc::new(Buffer::from_box(gl, materials)),
            time_id: 0
        }
    }

    pub unsafe fn mirror(&self) -> Self {
        Particles {
            buf: Buffer::<[_],_>::uninitialized(&self.buf.gl_provider(), self.buf.len()),
            solids: Buffer::<[_],_>::uninitialized(&self.solids.gl_provider(), self.solids.len()),
            boundary: self.boundary.clone(),
            materials: self.materials.clone(),
            time_id: self.time_id
        }
    }

    pub fn boundary(&self) -> &ParticleBuffer { &self.boundary }
    pub fn boundary_weak(&self) -> Weak<ParticleBuffer> {Rc::downgrade(&self.boundary.clone())}
    pub fn particles(&self) -> &ParticleBuffer { &self.buf }
    pub fn particles_mut(&mut self) -> &mut ParticleBuffer { &mut self.buf }
    pub fn solids(&self) -> &SolidParticleBuffer { &self.solids }
    pub fn solids_mut(&mut self) -> &mut SolidParticleBuffer { &mut self.solids }
    pub fn materials(&self) -> &Materials { &self.materials }

    pub fn add_particles(&mut self, material: Material, mut particles: Box<[Particle]>) {
        if particles.len()==0 {return;}

        let mat_id = {
            if material.normal_stiffness!=0.0 || material.shear_stiffness!=0.0 {
                self.materials.len()
            } else {
                let mut i = 0;
                let len = self.materials.len();
                for mat in self.materials.map().iter() {
                    if material == *mat {
                        break;
                    }
                    i += 1;
                }
                i
            }

        };

        for x in particles.iter_mut() {
            x.mat = mat_id as GLuint;
        }

        //add the new material, if it needs to be added
        if mat_id == self.materials.len() {
            unsafe {
                let new_mat = add_to_buffer(self.materials(), Box::new([material]));
                self.time_id += 1; //make this material list the new global one
                self.materials = Rc::new(new_mat);
            }
        }

        let solid_particles = Self::init_solid_particles(&mut particles, &self.materials().map(), self.solids.len(), self.buf.len());

        unsafe {
            use gl_struct::gl;

            //add the particles to the buffer

            let new_buf = add_to_buffer(&self.buf, particles);
            self.buf = new_buf;

            //add the solid particles to the buffer

            let new_buf = add_to_buffer(&self.solids, solid_particles);
            self.solids = new_buf;

        }

    }

}
