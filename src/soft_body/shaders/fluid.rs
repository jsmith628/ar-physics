use super::*;

glsl!{$
    pub mod fluid_forces {
        @Rust
            use super::*;
            use super::{Index};
            use super::kernel::*;
            use super::kernel::{kernel};

        @Compute
            #version 460

            #define EQ_ZERO 0
            #define EQ_CONSTANT 1
            #define EQ_IDEAL_GAS 2
            #define EQ_TAIT 3
            #define EQ_LENNARD_JONES 4
            #define EQ_HERTZ 5

            #define I (mat4(vec4(1,0,0,0),vec4(0,1,0,0),vec4(0,0,1,0),vec4(0,0,0,1)))
            #define ZERO (mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0)))
            #define EPSILON 0.1
            #define INVALID_INDEX 0xFFFFFFFF

            layout(local_size_x = 9, local_size_y = 3, local_size_z = 3) in;

            extern struct AABB;
            extern struct Material;
            extern struct MatInteraction;
            extern struct Particle;
            extern struct Bucket;
            extern struct Index;

            extern uint bucket_index(uvec3 pos, uvec4 sub);
            extern bool in_bounds(ivec3 b_pos, uvec4 sub);

            extern float normalization_constant(int n, float h);
            extern float kernel(vec4 r, float h, float k);
            extern vec4 grad_w(vec4 r, float h, float k);

            uniform int dim, mode;
            uniform float norm_const, h, g, f;

            float pressure(uint eq, float density, float temperature, float c, float d0) {
                switch(eq) {
                    case EQ_CONSTANT: return c*c;
                    case EQ_IDEAL_GAS: return c*c*(density - d0);
                    case EQ_TAIT: return (c*c*d0/7)*(pow(density/d0,7)-1);
                    default: return 0;
                }
            }

            float contact_potential(uint eq, float r, float strength, float r0) {
                switch(eq) {
                    case EQ_CONSTANT: return r<=r0 ? strength : 0;
                    case EQ_IDEAL_GAS: return strength*(r0-r);
                    case EQ_TAIT: return (strength*r/7)*(pow(r0/r,7)-1);
                    case EQ_LENNARD_JONES:
                        float factor = r0/r;
                        factor *= factor * factor;
                        factor *= factor;
                        return strength*(factor*factor - 2*factor);
                    case EQ_HERTZ: return (strength*r0/7)*(pow(r/r0,7)-1);
                    default: return 0;
                }
            }

            layout(std430) buffer particle_list { readonly restrict Particle particles[]; };
            layout(std430) buffer boundary_list { readonly restrict Particle boundary[]; };
            layout(std430) buffer derivatives { writeonly restrict Particle forces[]; };

            layout(std430) buffer material_list { readonly restrict Material materials[]; };
            layout(std430) buffer interaction_list { readonly restrict MatInteraction interactions[]; };

            layout(std430) buffer index_list { readonly restrict Index indices[]; };
            layout(std430) buffer neighbor_list {
                readonly restrict AABB bounds;
                readonly restrict uvec4 subdivisions;
                readonly restrict Bucket buckets[];
            };

            extern uint particle_index(uint bucket, uint i);

            const uint gids = gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z;
            shared float shared_den[gids];
            shared vec4 shared_force[gids];

            ivec3 local_bucket_pos(uint id) {
                ivec3 index = ivec3(buckets[id].index.xyz);
                ivec3 local_offset = ivec3(gl_LocalInvocationID.x%3, gl_LocalInvocationID.yz);
                return local_offset + index + ivec3(-1,-1,-1);
            }

            void main() {

                //get ids
                uint p_id = gl_WorkGroupID.x;
                uint s_id = particles[p_id].solid_id;
                uint gid = gl_LocalInvocationIndex;
                uint mat_id = particles[p_id].mat;

                //save these properties for convenience
                float m1 = materials[mat_id].mass;
                float d1 = particles[p_id].den;
                bool elastic = s_id!=INVALID_INDEX && (materials[mat_id].normal_stiffness!=0 || materials[mat_id].shear_stiffness!=0);

                //local variables for storing the result
                float den = 0;
                vec4 force = vec4(0,0,0,0);

                //ids for splitting the for loops
                uint sublocal_id = gl_LocalInvocationID.x/3;
                uint num_sublocal = gl_WorkGroupSize.x/3;

                //get this particle's bucket and offset to a neighboring bucket depending on our local id
                ivec3 b_pos = local_bucket_pos(indices[p_id].pos_index);

                //fluid forces
                if(in_bounds(b_pos, subdivisions)) {

                    //save these properties for convenience
                    uint state_eq = materials[mat_id].state_eq;
                    float c1 = materials[mat_id].sound_speed;
                    float f1 = materials[mat_id].visc;
                    float bf1 = materials[mat_id].bulk_visc;
                    float d0 = materials[mat_id].target_den;
                    float p1 = pressure(state_eq, d1, 0, c1, d0);

                    //get this neighbor's bucking index and list and boundary count
                    uint bucket_id = bucket_index(uvec3(b_pos), subdivisions);
                    uint bc = buckets[bucket_id].count[0];
                    uint ref_count = buckets[bucket_id].count[1];
                    uint count = buckets[bucket_id].count[2];

                    //add up the fluid force and density contributions from each particle
                    for(uint j=sublocal_id; j<count+ref_count+bc; j+=num_sublocal){
                        if(j>=bc && j<ref_count+bc) continue;

                        uint p_id2 = particle_index(bucket_id, j);
                        uint s_id2 = particles[p_id2].solid_id;
                        uint mat_2 = j<bc ? boundary[p_id2].mat : particles[p_id2].mat;

                        uint state_eq2 = materials[mat_2].state_eq;
                        float m2 = materials[mat_2].mass;
                        float f2 = materials[mat_2].visc;
                        float bf2 = materials[mat_2].bulk_visc;
                        float c2 = materials[mat_2].sound_speed;

                        vec4 r,v;
                        float d2;

                        if(j<bc) {
                            r = boundary[p_id2].pos - particles[p_id].pos;
                            v = boundary[p_id2].vel - particles[p_id].vel;
                            d2 = boundary[p_id2].den;
                            // d2 = boundary[p_id2].den + d1;
                            // c2 = 5000;
                            // state_eq2 = state_eq==EQ_ZERO ? EQ_TAIT : state_eq;
                        } else {
                            r = particles[p_id2].pos - particles[p_id].pos;
                            v = particles[p_id2].vel - particles[p_id].vel;
                            d2 = particles[p_id2].den;
                        }
                        float p2 = pressure(state_eq2, d2, 0, c2, materials[mat_2].target_den);

                        //density update
                        if(j>=bc) {
                        // if(mat_id==mat_2) {
                            den += m2 * dot(v, grad_w(r, h, norm_const));
                        }

                        vec4 contact_force = vec4(0,0,0,0);
                        bool elastic2 = s_id2!=INVALID_INDEX && (materials[mat_2].normal_stiffness!=0 || materials[mat_2].shear_stiffness!=0);

                        MatInteraction inter = interactions[mat_id*materials.length() + mat_2];
                        float contact_pressure = contact_potential(inter.potential, length(r), inter.radius, inter.strength);
                        // MatInteraction inter = interactions[0];
                        // float contact_pressure = 0;

                        //pressure force

                        force += m1*m2*(
                            p1/(d1*d1) +
                            p2/(d2*d2)
                        ) * grad_w(r, h, norm_const);

                        if(contact_pressure!=0){
                            contact_force += (m2*contact_pressure/d2) * grad_w(r, h, norm_const);

                            //friction
                            if((elastic || elastic2 || j<bc) && mat_id != mat_2) {
                                vec4 r_inv = r / dot(r,r);
                                vec4 normal_force = r_inv * dot(contact_force, r);
                                vec4 tangent_vel = v - r_inv* dot(v, r);
                                tangent_vel = normalize(tangent_vel);
                                if(!any(isnan(tangent_vel)) && !any(isinf(tangent_vel)))
                                    force += (inter.friction) * length(normal_force) * normalize(tangent_vel);
                            }

                            force += contact_force;
                        }

                        //viscocity
                        if(!elastic && j>=bc) force -= m1*m2*(f1+f2)*dot(r,grad_w(r, h, norm_const))*v/(d1*d2*(dot(r,r)+EPSILON*h*h));


                        //artificial viscocity
                        if(j>=bc) {
                            float _f = mat_id==mat_2 ? bf1+bf2 : inter.dampening;
                            force -= m1*m2*(
                                (_f)*dot(v, r) / ((d1+d2)*(dot(r,r)+EPSILON*h*h))
                            )*grad_w(r, h, norm_const);
                        }
                    }

                }


                //store the contribution of this local id in a shared array
                shared_den[gid] = den;
                shared_force[gid] = force;
                barrier();

                //next, sum up all local contributions, apply body forces and shift the velocity
                uint shadow = gids;
                while(shadow%3==0) {
                    shadow /= 3;
                    if(gid<shadow) {
                        shared_den[gid] += shared_den[gid+shadow] + shared_den[gid+shadow*2];
                        shared_force[gid] += shared_force[gid+shadow] + shared_force[gid+shadow*2];
                    }
                    barrier();
                }

                if(gid==0) {
                    //add up each local unit's contributions
                    for(uint i=1; i<shadow; i++) {
                        shared_den[0] += shared_den[i];
                        shared_force[0] += shared_force[i];
                    }

                    forces[p_id].mat = mat_id;
                    forces[p_id].solid_id = s_id;
                    forces[p_id].den = shared_den[0];
                    forces[p_id].vel = shared_force[0]/particles[p_id].den + vec4(0,-g,0,0);
                    forces[p_id].pos = particles[p_id].vel;

                }

            }
    }
}
