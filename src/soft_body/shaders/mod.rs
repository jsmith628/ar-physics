use super::*;

pub use fluid::*;
pub use solid::*;
pub use strain::*;

mod fluid;
mod solid;
mod strain;

pub fn compute_forces(
    fluid_force: &mut fluid_forces::Program,
    solid_force: &mut solid_forces::Program,
    strain: &mut compute_strain::Program,
    clear_solids: &mut clear_solids::Program,
    buckets: &mut NeighborList,
    particles: ParticleState
) -> ParticleState {

    #[allow(mutable_transmutes)]
    particles.map_terms(
        |p| unsafe {
            let prof = crate::PROFILER.as_mut().unwrap();
            prof.new_segment("Bucketting".to_owned());

            let (indices, buckets) = buckets.update_contents(&p);

            let particles = p.particles();
            let solids = p.solids();
            let materials = p.materials();
            let interactions = p.interactions();

            //NOTE: we know for CERTAIN that neither of these are modified by the shader,
            //so against all warnings, we are going to transmute them to mutable

            let ub_mat: &mut MaterialBuffer = ::std::mem::transmute::<&MaterialBuffer,&mut MaterialBuffer>(materials);
            let ub_inter: &mut InteractionBuffer = ::std::mem::transmute::<&InteractionBuffer,&mut InteractionBuffer>(interactions);
            let ub = ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(&*particles);
            let ub_bound = ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(p.boundary());

            prof.new_segment("Forces".to_owned());

            let mut dest = p.mirror();

            fluid_force.compute(
                p.particles().len() as u32, 1, 1,
                ub, ub_bound, dest.particles_mut(),
                ub_mat, ub_inter, indices, buckets
            );

            if solids.len() > 1 {

                clear_solids.clear_solids(&solids, dest.solids_mut());

                let mut dest2 = p.mirror();
                let mut strains = Buffer::<[[mat4;3]],Read>::uninitialized(&particles.gl_provider(), solids.len());

                let ub_s = ::std::mem::transmute::<&SolidParticleBuffer,&mut SolidParticleBuffer>(&*solids);
                strain.compute(strains.len() as u32, 1, 1, ub, ub_s, ub_mat, indices, &mut strains, buckets);
                let (dest_p, dest_s) = dest2.all_particles_mut();

                solid_force.compute(
                    p.particles().len() as u32, 1, 1,
                    ub, ub_s, ub_bound, dest_p, dest_s,
                    ub_mat, &mut strains,
                    indices, buckets
                );

                // let solids_map = solids.map();
                // for (sf, sp) in dest.solids_mut().map_mut().iter_mut().zip(solids_map.iter()) {
                //     sf.part_id = sp.part_id;
                //     sf.ref_pos = [0.0,0.0,0.0,0.0].into();
                //     sf.stress = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]].into();
                // }
                // drop(solids_map);

                prof.end_segment();

                vec![dest, dest2]
            } else {
                vec![dest]
            }

        }
    )
}
