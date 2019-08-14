use super::*;

glsl!{$
    
    pub mod compute_strain {

        @Rust
            use super::*;
            use super::{Index};
            use super::kernel::*;

        @Compute
            #version 460

            #define I (mat4(vec4(1,0,0,0),vec4(0,1,0,0),vec4(0,0,1,0),vec4(0,0,0,1)))
            #define ZERO (mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0)))

            layout(local_size_x = 9, local_size_y = 3, local_size_z = 3) in;

            extern struct AABB;
            extern struct Material;
            extern struct Particle;
            extern struct SolidParticle;
            extern struct Bucket;
            extern struct Index;

            extern uint bucket_index(uvec3 pos, uvec4 sub);
            extern bool in_bounds(ivec3 b_pos, uvec4 sub);

            extern float normalization_constant(int n, float h);
            extern vec4 grad_w(vec4 r, float h, float k);

            layout(std430) buffer particle_list { readonly restrict Particle particles[]; };
            layout(std430) buffer solid_particle_list { readonly restrict SolidParticle solids[]; };
            layout(std430) buffer material_list { readonly restrict Material materials[]; };
            layout(std430) buffer index_list { readonly restrict Index indices[]; };

            layout(std430) buffer strain_list { writeonly restrict mat4 strains[][3]; };

            layout(std430) buffer neighbor_list {
                readonly restrict AABB bounds;
                readonly restrict uvec4 subdivisions;
                readonly restrict Bucket buckets[];
            };

            const uint gids = gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z;

            extern uint particle_index(uint bucket, uint i);

            ivec3 local_bucket_pos(uint id) {
                ivec3 index = ivec3(buckets[id].index.xyz);
                ivec3 local_offset = ivec3(gl_LocalInvocationID.x%3, gl_LocalInvocationID.yz);
                return local_offset + index + ivec3(-1,-1,-1);
            }

            shared mat4 deformation[gids];
            shared mat4 def_rate[gids];
            shared mat4 correction[gids];

            bool is_elastic(uint id) {
                uint mat = particles[id].mat;
                return materials[mat].normal_stiffness!=0 || materials[mat].shear_stiffness!=0;
            }

            uniform uint dim;
            uniform float norm_const, h;

            // extern mat4 strain_measure(mat4 def_grad);

            void main() {

                uint s_id = gl_WorkGroupID.x;
                uint p_id = solids[s_id].part_id;
                uint gid = gl_LocalInvocationIndex;

                if(gid==0) {
                    strains[s_id][0] = ZERO;
                    strains[s_id][1] = ZERO;
                    strains[s_id][2] = ZERO;
                }
                barrier();

                //if this isn't an elastic particle, leave
                if(is_elastic(p_id)) {

                    //ids for splitting the for loops
                    uint sublocal_id = gl_LocalInvocationID.x/3;
                    uint num_sublocal = gl_WorkGroupSize.x/3;

                    correction[gid] = ZERO;
                    deformation[gid] = ZERO;
                    def_rate[gid] = ZERO;

                    ivec3 b_pos = local_bucket_pos(indices[p_id].ref_index);
                    bool in_bounds = in_bounds(b_pos, subdivisions);

                    if(in_bounds) {
                        uint b_id = bucket_index(uvec3(b_pos), subdivisions);
                        uint start = buckets[b_id].count[0];
                        uint end = start + buckets[b_id].count[1];

                        //first, we need to compute the correction
                        for(uint i=start+sublocal_id; i<end; i+=num_sublocal) {
                            uint p_id2 = particle_index(b_id, i);
                            uint s_id2 = particles[p_id2].solid_id;
                            if(is_elastic(p_id2)) {
                                float V2 = materials[particles[p_id2].mat].mass / particles[p_id2].den;
                                vec4 dX = solids[s_id2].ref_pos - solids[s_id].ref_pos;

                                correction[gid] += V2*outerProduct(grad_w(dX, h, norm_const), dX);
                            }
                        }
                    }
                    barrier();

                    //next, sum up all local contributions
                    uint shadow = gids;
                    while(shadow%3==0) {
                        shadow /= 3;
                        if(gid<shadow) {
                            correction[gid] += correction[gid+shadow] + correction[gid+shadow*2];
                        }
                        barrier();
                    }

                    //invert the correction matrix
                    if(gid==0) {
                        for(uint i=dim; i<4; i++) correction[0][i][i] = 1.0;
                        correction[0] = inverse(correction[0]);
                    }
                    barrier();

                    if(in_bounds) {
                        uint b_id = bucket_index(uvec3(b_pos), subdivisions);
                        uint start = buckets[b_id].count[0];
                        uint end = start + buckets[b_id].count[1];

                        //next, we get the gradients to compute the correction
                        for(uint i=start+sublocal_id; i<end; i+=num_sublocal) {
                            uint p_id2 = particle_index(b_id, i);
                            uint s_id2 = particles[p_id2].solid_id;
                            if(!is_elastic(p_id2)) continue;

                            float V2 = materials[particles[p_id2].mat].mass / particles[p_id2].den;
                            vec4 dX = solids[s_id2].ref_pos - solids[s_id].ref_pos;
                            vec4 dx = particles[p_id2].pos - particles[p_id].pos;
                            vec4 dv = particles[p_id2].vel - particles[p_id].vel;
                            vec4 grad = correction[0] * grad_w(dX, h, norm_const);

                            deformation[gid] += V2 * outerProduct(dx, grad);
                            def_rate[gid] += V2 * outerProduct(dv, grad);
                        }
                    }
                    barrier();

                    //next, sum up all local contributions
                    shadow = gids;
                    while(shadow%3==0) {
                        shadow /= 3;
                        if(gid<shadow) {
                            deformation[gid] += deformation[gid+shadow] + deformation[gid+shadow*2];
                            def_rate[gid] += def_rate[gid+shadow] + def_rate[gid+shadow*2];
                        }
                        barrier();
                    }

                    if(gid==0) {
                        strains[s_id][0] = correction[0];
                        strains[s_id][1] = deformation[0];
                        strains[s_id][2] = def_rate[0];
                    }

                }

            }


    }
}
