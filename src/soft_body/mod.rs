
use gl_struct::glsl_type::*;
use gl_struct::*;

use std::cell::*;

use self::neighbor_list::*;
use numerical_integration::*;
use self::material_region::*;
// use self::particle_list::*;
// use self::particle_list::Particle;
use self::particle_state::*;

// pub mod particle_list;
pub mod particle_state;
pub mod neighbor_list;
pub mod material_region;
pub mod kernel;

use shaders::*;
mod shaders;

pub struct FluidSim {
    //integration
    integrator: Box<dyn VelIntegrates<f32, ParticleState>>,
    timestep: f32,
    subticks: uint,

    //state
    time: f32,
    particles: Particles,
    state: Box<[ParticleState]>,

    //data for update logistics
    neighbor_list: RefCell<NeighborList>,

    //shaders
    fluid_forces: RefCell<fluid_forces::Program>,
    solid_forces: RefCell<solid_forces::Program>,
    strain: RefCell<compute_strain::Program>

}

impl FluidSim {

    pub fn with_integrator<I: VelIntegrator+Sized+'static>(
        gl: &GLProvider,
        fluids: &[MaterialRegion],
        interactions: &[&[Option<MatInteraction>]],
        bounds: AABB, kernel_rad: f32,
        integrator: I, timestep: f32, subticks: uint,
        gravity: f32, artificial_viscocity: f32,
    ) -> Result<Self, GLError> {
        Self::new(gl, fluids, interactions, bounds, kernel_rad, Box::new(integrator), timestep, subticks, gravity, artificial_viscocity)
    }

    pub fn new(
        gl: &GLProvider,
        fluids: &[MaterialRegion],
        interactions: &[&[Option<MatInteraction>]],
        bounds: AABB, kernel_rad: f32,
        integrator: Box<dyn VelIntegrates<f32, ParticleState>>,
        timestep: f32, subticks: uint,
        gravity: f32, artificial_viscocity: f32,
    ) -> Result<Self, GLError> {

        unsafe { crate::PROFILER = Some(crate::profiler::Profiler::new()) };

        let mut boundary = Vec::new();
        let mut particles = Vec::new();
        let mut materials = Vec::new();
        for i in 0..fluids.len() {
            let fluid = &fluids[i];
            let (p, mat) = fluid.gen_particles(kernel_rad, i as u32);
            if mat.immobile.into() {
                boundary.push(p);
            } else {
                particles.push(p);
            }
            materials.push(mat);

        }

        let mut boundary: Vec<_> = boundary.into_iter().flatten().collect();
        let mut particles: Vec<_> = particles.into_iter().flatten().collect();


        boundary.shrink_to_fit();
        particles.shrink_to_fit();
        materials.shrink_to_fit();

        let mut inter = vec![MatInteraction::default(); materials.len()*materials.len()].into_boxed_slice();
        for i in 0..materials.len() {
            for j in i..materials.len() {
                let interaction = {
                    if let Some(mi) = interactions[i][j] {
                        mi
                    } else if let Some(mi) = interactions[j][i] {
                        mi
                    } else if i!=j {
                        MatInteraction::default_between(materials[i], materials[j], kernel_rad)
                    } else {
                        MatInteraction::default()
                    }
                };
                println!("{:?}", interaction);
                inter[i*materials.len() + j] = interaction;
                inter[j*materials.len() + i] = interaction;
            }
        }

        let dim = {
            let mut d = 0;
            for i in 0..4 { if bounds.dim[i] > 0.0 {d += 1u32}; }
            d
        };

        println!("dim={}\nboundary particles: {}\nmobile particles: {}", dim, boundary.len(), particles.len());

        let fs = FluidSim {
            integrator: integrator,
            timestep: timestep,
            subticks: subticks,

            time: 0.0,
            particles: Particles::new(gl, (materials.into_boxed_slice(),inter), particles.into_boxed_slice(), boundary.into_boxed_slice()),
            state: Vec::new().into_boxed_slice(),

            neighbor_list: RefCell::new(NeighborList::new(gl, bounds, kernel_rad)),

            fluid_forces: RefCell::new(fluid_forces::init(gl).unwrap()),
            solid_forces: RefCell::new(solid_forces::init(gl).unwrap()),
            strain: RefCell::new(compute_strain::init(gl).unwrap()),
        };

        use self::kernel::norm_const;
        let mut fluid_forces = fs.fluid_forces.borrow_mut();
        let mut solid_forces = fs.solid_forces.borrow_mut();
        let mut strain = fs.strain.borrow_mut();

        *fluid_forces.dim = dim as i32;
        *fluid_forces.norm_const = norm_const(dim, kernel_rad);
        *fluid_forces.h = kernel_rad;
        *fluid_forces.f = artificial_viscocity;
        *fluid_forces.g = gravity;

        *solid_forces.dim = dim as i32;
        *solid_forces.norm_const = norm_const(dim, kernel_rad);
        *solid_forces.h = kernel_rad;
        *solid_forces.f = artificial_viscocity;
        *solid_forces.g = gravity;

        *strain.dim = dim as u32;
        *strain.h = kernel_rad;
        *strain.norm_const = norm_const(dim, kernel_rad);

        drop(fluid_forces);
        drop(solid_forces);
        drop(strain);

        Ok(fs)
    }

    pub fn kernel_radius(&self) -> f32 { *self.fluid_forces.borrow().h }

    pub fn time(&self) -> f32 {self.time}

    pub fn particles(&self) -> &Particles {
        &self.particles
    }

    pub fn add_particles(&mut self, obj: MaterialRegion, offset: Option<vec4>) {
        let h = *self.fluid_forces.borrow().h;
        let (mut p, mat) = obj.gen_particles(h, 0);

        if let Some(t) = offset {
            for x in p.iter_mut() {
                x.pos[0] += t[0];
                x.pos[1] += t[1];
                x.pos[2] += t[2];
                x.pos[3] += t[3];
            }
        }

        self.state[0].map_mut(|particles| particles.add_particles(mat, p.into_boxed_slice(), h));
        self.particles = self.state[0].clone().map_into(|p| p).unwrap();

    }

}

impl ::ar_engine::engine::Component for FluidSim {

    #[inline]
    fn init(&mut self) {
        let neighbors = &self.neighbor_list;
        let ff = &self.fluid_forces;
        let sf = &self.solid_forces;
        let strains = &self.strain;

        self.state = self.integrator.init_with_vel(
            ParticleState::new(self.particles.clone()),
            self.timestep / self.subticks as f32,
            & |_t, state| state.velocity(),
            & |_t, state| compute_forces(
                &mut ff.borrow_mut(),
                &mut sf.borrow_mut(),
                &mut strains.borrow_mut(),
                &mut neighbors.borrow_mut(),
                state
            )
        );
    }

    fn update(&mut self) {
        let prof = unsafe { crate::PROFILER.as_mut().unwrap() };
        println!("{:?}", prof.new_frame());

        let dt = self.timestep / self.subticks as f32;
        let mut state = None;
        for _ in 0..self.subticks {
            self.time += dt;

            let neighbors = &self.neighbor_list;
            let ff = &self.fluid_forces;
            let sf = &self.solid_forces;
            let strains = &self.strain;

            state = Some(
                self.integrator.step_with_vel(
                    self.time,
                    self.state.as_mut(),
                    dt,
                    & |_t, state| state.velocity(),
                    & |_t, state| compute_forces(
                        &mut ff.borrow_mut(),
                        &mut sf.borrow_mut(),
                        &mut strains.borrow_mut(),
                        &mut neighbors.borrow_mut(),
                        state
                    )
                )
            );

        }

        self.particles = state.unwrap().map_into(|p| p).unwrap();

        prof.new_segment("Graphics".to_owned());

    }
}
