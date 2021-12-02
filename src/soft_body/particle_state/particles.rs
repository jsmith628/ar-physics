
macro_rules! count {
    () => { 0 };
    ($t0:tt $($tt:tt)*) => { 1 + count!($($tt)*) };
}


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
                            const N: usize = count!($($r)*);
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
                        if(id >= dst.length()) return;

                        if(id < $p0.length()) {
                            dst[id].mat = $p0[id].mat;
                            dst[id].solid_id = $p0[id].solid_id;
                        }
                        $(else if(id < $p2.length()) {
                            dst[id].mat = $p2[id].mat;
                            dst[id].solid_id = $p2[id].solid_id;
                        })*

                        dst[id].den = (id<$p0.length() ? $r0*$p0[id].den : 0) $(+(id<$p2.length() ? $r2*$p2[id].den : 0))*;
                        // dst[id].ref_pos = (id<$p0.length() ? $r0*$p0[id].ref_pos : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].ref_pos : VZERO))*;
                        dst[id].pos = (id<$p0.length() ? $r0*$p0[id].pos : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].pos : VZERO))*;
                        dst[id].vel = (id<$p0.length() ? $r0*$p0[id].vel : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].vel : VZERO))*;
                        // dst[id].stress = (id<$p0.length() ? $r0*$p0[id].stress : MZERO) $(+(id<$p2.length() ? $r2*$p2[id].stress : MZERO))*;

                    }
            }

            mod $shdr2 {
                @Rust
                    use super::*;
                    use std::mem::transmute;
                    use std::cell::RefCell;

                    impl Program {
                        pub(super) fn into_closure(self) -> SolidLCClosure {
                            const N: usize = count!($($r)*);
                            let shader = RefCell::new(self);
                            Box::new(
                                move |arr| {
                                    let (split, _) = arr.split_at(N);
                                    let len = split.iter().fold(0, |l,p| l.max(p.1.len())) as GLuint;
                                    let gl = split[0].1.gl_provider();
                                    if let [$(($r, $p)),*] = split {
                                        unsafe {
                                            let mut dest = Buffer::<[_],_>::uninitialized(&gl, len as usize);
                                            let [$($p_alt),*] = transmute::<[&SolidParticleBuffer;N], [&mut SolidParticleBuffer;N]>([$($p),*]);

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

                    extern struct SolidParticle;

                    $(uniform float $r = 0.0;)*
                    $(layout(std430) buffer $p_alt {readonly SolidParticle $p[];};)*
                    layout(std430) buffer dest {writeonly restrict SolidParticle dst[];};

                    void main() {
                        uint id = gl_GlobalInvocationID.x;
                        if(id >= dst.length()) return;

                        if(id < $p0.length()) {
                            dst[id].part_id = $p0[id].part_id;
                        }
                        $(else if(id < $p2.length()) {dst[id].part_id = $p2[id].part_id;})*

                        dst[id].ref_pos = (id<$p0.length() ? $r0*$p0[id].ref_pos : VZERO) $(+(id<$p2.length() ? $r2*$p2[id].ref_pos : VZERO))*;
                        dst[id].stress = (id<$p0.length() ? $r0*$p0[id].stress : MZERO) $(+(id<$p2.length() ? $r2*$p2[id].stress : MZERO))*;

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
                if(id < part.length() && id < dest.length()) {
                    dest[id].mat = part[id].mat;
                    dest[id].solid_id = part[id].solid_id;
                    dest[id].den = 0.0;
                    dest[id].pos = r1*part[id].vel;
                    dest[id].vel = vec4(0.0,0.0,0.0,0.0);
                }
            }
    }

    mod vel_solid {
        @Rust
            use super::*;
            use std::mem::transmute;
            use std::cell::RefCell;

            impl Program {
                pub(super) fn into_closure(self) -> Box<dyn for<'a> Fn(&'a SolidTerm<'a>) -> SolidParticleBuffer> {
                    let shader = RefCell::new(self);
                    Box::new(
                        move |(r, p)| {
                            let mut shdr = shader.borrow_mut();
                            *shdr.r1 = *r;

                            #[allow(mutable_transmutes)]
                            unsafe {
                                let mut dest = Buffer::<[_],_>::uninitialized(&p.gl_provider(), p.len());
                                let p_mut = transmute::<&SolidParticleBuffer, &mut SolidParticleBuffer>(p);
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

            extern struct SolidParticle;

            uniform float r1 = 1.0;
            layout(std430) buffer part_ {readonly restrict SolidParticle part[];};
            layout(std430) buffer dest_ {writeonly restrict SolidParticle dest[];};

            void main() {
                uint id = gl_GlobalInvocationID.x;
                if(id < part.length() && id < dest.length()) {
                    dest[id].part_id = part[id].part_id;
                    dest[id].ref_pos = vec4(0.0,0.0,0.0,0.0);
                    dest[id].stress = mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0));
                }
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

gen_lin_comb!($ lin_comb_1 s_lin_comb_1 (r1 p1 _p1) );
gen_lin_comb!($ lin_comb_2 s_lin_comb_2 (r1 p1 _p1) (r2 p2 _p2) );
gen_lin_comb!($ lin_comb_3 s_lin_comb_3 (r1 p1 _p1) (r2 p2 _p2) (r3 p3 _p3) );
gen_lin_comb!($ lin_comb_4 s_lin_comb_4 (r1 p1 _p1) (r2 p2 _p2) (r3 p3 _p3) (r4 p4 _p4) );
gen_lin_comb!($ lin_comb_5 s_lin_comb_5 (r1 p1 _p1) (r2 p2 _p2) (r3 p3 _p3) (r4 p4 _p4) (r5 p5 _p5) );

use super::*;
use std::rc::Rc;
use maths_traits::analysis::ComplexSubset;

pub(self) use super::Particle;
pub(self) fn units(p: GLuint) -> GLuint { ComplexSubset::ceil((p) as GLfloat / 128.0) as GLuint }

pub(self) type Term<'a> = (GLfloat, &'a ParticleBuffer);
pub(self) type SolidTerm<'a> = (GLfloat, &'a SolidParticleBuffer);
pub(self) type CombinedTerm<'a> = (GLfloat, &'a ParticleBuffer, &'a SolidParticleBuffer);
pub(self) type OwnedTerm = (GLfloat, ParticleBuffer, SolidParticleBuffer);
pub(self) type LCClosure = Box<dyn for<'a> Fn(&'a [Term]) -> ParticleBuffer>;
pub(self) type SolidLCClosure = Box<dyn for<'a> Fn(&'a [SolidTerm]) -> SolidParticleBuffer>;

#[derive(Clone)]
pub struct ArithShaders {
    lc: Rc<[LCClosure]>,
    slc: Rc<[SolidLCClosure]>,
    vel: Rc<dyn for<'a> Fn(&'a Term<'a>) -> ParticleBuffer>,
    vel_solid: Rc<dyn for<'a> Fn(&'a SolidTerm<'a>) -> SolidParticleBuffer>,
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
            slc: Rc::new(
                [
                    s_lin_comb_1::init(gl)?.into_closure(),
                    s_lin_comb_2::init(gl)?.into_closure(),
                    s_lin_comb_3::init(gl)?.into_closure(),
                    s_lin_comb_4::init(gl)?.into_closure(),
                    s_lin_comb_5::init(gl)?.into_closure(),
                ]
            ),
            vel: vel::init(gl)?.into_closure().into(),
            vel_solid: vel_solid::init(gl)?.into_closure().into(),
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

    fn reduce_owned(&self, mut terms: Vec<OwnedTerm>) -> Option<(ParticleBuffer,SolidParticleBuffer)> {
        // self.reduce(terms, Vec::new())
        if terms.len() == 0 { return None; }
        if terms.len() == 1 && terms[0].0 == 1.0 {
            let (_, p, s) = terms.pop().unwrap();
            return Some((p,s));
        }
        self.reduce_owned(
            terms.chunks(self.lc.len()).map(
                |arr|
                (1.0,
                    (self.lc[arr.len()-1])(
                        arr.iter().map(|(r, b, _)| (*r, b)).collect::<Vec<_>>().as_slice()
                    ),
                    (self.slc[arr.len()-1])(
                        arr.iter().map(|(r, _, b)| (*r, b)).collect::<Vec<_>>().as_slice()
                    ),
                )
            ).collect()
        )
    }

    fn reduce_ref<'a>(&self, terms: Vec<CombinedTerm<'a>>) -> Option<(ParticleBuffer,SolidParticleBuffer)> {
        if terms.len() == 0 { return None; }
        self.reduce_owned(
            terms.chunks(self.lc.len()).map(
                |arr| (1.0,
                    (self.lc[arr.len()-1])(
                        arr.iter().map(|(r, b, _)| (*r, *b)).collect::<Vec<_>>().as_slice()
                    ),
                    (self.slc[arr.len()-1])(
                        arr.iter().map(|(r, _, b)| (*r, *b)).collect::<Vec<_>>().as_slice()
                    ),
                )
            ).collect()
        )
    }

    pub fn linear_combination<'a>(&self, terms: Vec<(GLfloat, &'a Particles)>) -> Option<Particles> {
        self.reduce_ref(terms.iter().map(|(r,p)| (*r,p.particles(),p.solids())).collect()).map(
            |(p,s)| {

                let latest = terms.iter().fold(terms[0].1,
                    |m,p| if p.1.time_id > m.time_id {p.1} else {m}
                );

                Particles {
                    buf: p,
                    solids: s,
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
            solids: (self.vel_solid)(&(term.0, term.1.solids())),
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

    gl::MemoryBarrier(gl::ALL_BARRIER_BITS);

    let len = new_buf.len();
    new_buf.map_mut()[old.len()..len].copy_from_slice(&extra);

    new_buf
}

impl Particles {

    fn init_solid_particles(particles: &mut [Particle], materials: &[Material], offset:usize, p_offset:usize) -> Box<[SolidParticle]> {
        let mut result = Vec::new();
        let mut i = p_offset;
        let mut j = offset;

        for p in particles {
            if materials[p.mat as usize].is_solid() {
                p.solid_id = j as GLuint;
                result.push(SolidParticle::new(i as GLuint,p.pos));
                j += 1;
            }
            i += 1;
        }

        if result.len() == 0 { result.push(SolidParticle::new(0xFFFFFFFF,[0.0,0.0,0.0,0.0].into())) }
        result.shrink_to_fit();

        result.into_boxed_slice()
    }



    pub fn new(
        gl: &GLProvider,
        materials: (Box<[Material]>, Box<[MatInteraction]>),
        mut particles: Box<[Particle]>, boundary: Box<[Particle]>
    ) -> Self {
        Particles{
            solids: Buffer::from_box(gl, Self::init_solid_particles(&mut particles, &materials.0, 0,0)),
            buf: Buffer::from_box(gl, particles),
            boundary: Rc::new(Buffer::from_box(gl, boundary)),
            materials: Rc::new((Buffer::from_box(gl, materials.0), Buffer::from_box(gl, materials.1))),
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
    pub fn materials(&self) -> &MaterialBuffer { &self.materials.0 }
    pub fn interactions(&self) -> &InteractionBuffer { &self.materials.1 }

    pub fn all_particles_mut(&mut self) -> (&mut ParticleBuffer, &mut SolidParticleBuffer) {
        (&mut self.buf, &mut self.solids)
    }

    pub fn add_particles(&mut self, material: Material, mut particles: Box<[Particle]>, h:GLfloat) {
        if particles.len()==0 {return;}

        let mat_id = {
            if material.is_solid() {
                self.materials().len()
            } else {
                let mut i = 0;
                for mat in self.materials().map().iter() {
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
        if mat_id == self.materials().len() {

            unsafe {
                let new_mat = add_to_buffer(self.materials(), Box::new([material]));
                let (mat_buf, int_buf) = Rc::make_mut(&mut self.materials);
                *mat_buf = new_mat;

                //next, remake the interaction array
                let new_len = mat_buf.len();
                let old_len = new_len - 1;
                let old_interactions = int_buf.read_into_box();
                let mut new_interactions = Vec::with_capacity(new_len*new_len);

                let mats = mat_buf.map();
                for i in 0..old_len {
                    for j in 0..old_len {
                        new_interactions.push(old_interactions[i*old_len+j]);
                    }
                    new_interactions.push(MatInteraction::default_between(mats[i], mats[old_len], h));
                }
                for i in 0..old_len {
                    new_interactions.push(MatInteraction::default_between(mats[old_len], mats[i], h));
                }

                if material.is_solid() {
                    new_interactions.push(MatInteraction::default());
                } else {
                    new_interactions.push(MatInteraction::default_between(mats[old_len], mats[old_len], h));
                }

                drop(mats);

                new_interactions.shrink_to_fit();
                *int_buf = Buffer::from_box(&mat_buf.gl_provider(), new_interactions.into_boxed_slice());

                self.time_id += 1; //make this material list the new global one

            }
        }

        unsafe {

            //add the solid particles to the buffer
            if material.is_solid() {
                let sp = Self::init_solid_particles(
                    &mut particles, &self.materials().map(), self.solids.len(), self.buf.len()
                );
                let new_buf = add_to_buffer(&self.solids, sp);
                self.solids = new_buf;
            }

            //add the particles to the buffer
            let new_buf = add_to_buffer(&self.buf, particles);
            self.buf = new_buf;

            // println!("{:?}", self.solids.read_into_box().iter().map(|p|p.part_id).collect::<Vec<_>>());


        }

    }

}
