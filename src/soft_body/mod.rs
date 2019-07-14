
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

glsl!{$

    pub mod pressure_force {
        @Rust
            use super::*;
            use super::{Index};
            use super::kernel::*;
            use super::kernel::{kernel};

        @Compute
            #version 460

            #define I (mat4(vec4(1,0,0,0),vec4(0,1,0,0),vec4(0,0,1,0),vec4(0,0,0,1)))
            #define ZERO (mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0)))
            #define EPSILON 0.1

            layout(local_size_x = 9, local_size_y = 3, local_size_z = 3) in;

            extern struct AABB;
            extern struct Material;
            extern struct Particle;
            extern struct Bucket;
            extern struct Index;

            extern uint bucket_index(uvec3 pos, uvec4 sub);
            extern bool in_bounds(ivec3 b_pos, uvec4 sub);

            extern float normalization_constant(int n, float h);
            extern float kernel(vec4 r, float h, float k);
            extern vec4 grad_w(vec4 r, float h, float k);

            float pressure(uint eq, float density, float temperature, float c, float d0) {
                if(eq==0) {
                    return 0;
                } else if(eq==1) {
                    return state(density, temperature, c, d0);
                } else {
                    return ideal_gas(density, temperature, c, d0);
                }
            }

            layout(std430) buffer particle_list { readonly restrict Particle particles[]; };
            layout(std430) buffer boundary_list { readonly restrict Particle boundary[]; };
            layout(std430) buffer derivatives { restrict Particle forces[]; };

            layout(std430) buffer material_list { readonly restrict Material materials[]; };

            layout(std430) buffer index_list { readonly restrict Index indices[]; };
            layout(std430) buffer neighbor_list {
                readonly restrict AABB bounds;
                readonly restrict uvec4 subdivisions;
                readonly restrict Bucket buckets[];
            };

            extern uint particle_index(uint bucket, uint i);

            void main() {
                //get ids
                uint id = gl_WorkGroupID.x;
                uint gid = gl_LocalInvocationIndex;
                uint mat_id = particles[id].mat;

                //save these properties for convenience
                float m1 = materials[mat_id].mass;
                float d1 = particles[id].den;
                bool elastic = materials[mat_id].normal_stiffness!=0 || materials[mat_id].shear_stiffness!=0;

                //local variables for storing the result
                float den = 0;
                vec4 force = vec4(0,0,0,0);

                //ids for splitting the for loops
                uint sublocal_id = gl_LocalInvocationID.x/3;
                uint num_sublocal = gl_WorkGroupSize.x/3;

                //get this particle's bucket and offset to a neighboring bucket depending on our local id
                ivec3 b_pos = local_bucket_pos(indices[id].pos_index);

                //fluid forces
                if(in_bounds(b_pos, subdivisions)) {

                    //save these properties for convenience
                    uint state_eq = materials[mat_id].state_eq;
                    float c1 = materials[mat_id].sound_speed;
                    float f1 = materials[mat_id].visc;
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

                        uint id2 = particle_index(bucket_id, j);
                        uint mat_2 = j<bc ? boundary[id2].mat : particles[id2].mat;

                        uint state_eq2 = materials[mat_2].state_eq;
                        float m2 = materials[mat_2].mass;
                        float f2 = materials[mat_2].visc;
                        float c2 = materials[mat_2].sound_speed;

                        vec4 r,v;
                        float d2;

                        if(j<bc) {
                            r = boundary[id2].pos - particles[id].pos;
                            v = boundary[id2].vel - particles[id].vel;
                            d2 = boundary[id2].den + d1;
                            c2 = 5000;
                            state_eq2 = state_eq==0 ? 1 : state_eq;
                        } else {
                            r = particles[id2].pos - particles[id].pos;
                            v = particles[id2].vel - particles[id].vel;
                            d2 = particles[id2].den;
                        }
                        float p2 = pressure(state_eq2, d2, 0, c2, materials[mat_2].target_den);

                        //density update
                        if(!elastic || mat_id==mat_2) {
                            den += m2 * dot(v, grad_w(r, h, norm_const));
                        }

                        vec4 contact_force = vec4(0,0,0,0);
                        bool elastic2 = materials[mat_2].normal_stiffness!=0 || materials[mat_2].shear_stiffness!=0;

                        //pressure force
                        if(!elastic || !elastic2) {
                            vec4 pressure_force = m1*m2*(
                                p1/(d1*d1) +
                                p2/(d2*d2)
                            ) * grad_w(r, h, norm_const);
                            force += pressure_force;
                            if(mat_id != mat_2) contact_force += pressure_force;
                        }

                        //viscocity
                        if(!elastic) force += m1*m2*(f1+f2)*dot(r,grad_w(r, h, norm_const))*v/(d1*d2*(dot(r,r)+EPSILON*h*h));

                        //friction
                        if((elastic || elastic2 || j<bc) && mat_id != mat_2) {
                            vec4 r_inv = r / dot(r,r);
                            vec4 normal_force = r_inv * dot(contact_force, r);
                            vec4 tangent_vel = v - r_inv* dot(v, r);
                            force += (f1 + f2) * length(normal_force) * normalize(tangent_vel);
                        }

                        //artificial viscocity
                        if(j>=bc) force -= m1*m2*(
                            (f*h*(c1+c2))*dot(v, r) / ((d1+d2)*(dot(r,r)+EPSILON*h*h))
                        )*grad_w(r, h, norm_const);

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
                    memoryBarrierShared();
                }

                if(gid==0) {

                    forces[id].mat = mat_id;
                    forces[id].den = 0;
                    forces[id].vel = vec4(0,0,0,0);
                    forces[id].pos = particles[id].vel;

                    //add up each local unit's contributions
                    for(uint i=0; i<shadow; i++) {
                        forces[id].den += shared_den[i];
                        forces[id].vel += shared_force[i];
                    }

                    //get acceleration from force and gravity
                    forces[id].vel /= particles[id].den;
                    forces[id].vel += vec4(0,-g,0,0);
                }

            }


    }


    pub mod compute_force {
        @Rust
            use super::*;
            use super::{Index};
            use super::kernel::*;
            use super::kernel::{kernel};

        @Compute
            #version 460

            #define I (mat4(vec4(1,0,0,0),vec4(0,1,0,0),vec4(0,0,1,0),vec4(0,0,0,1)))
            #define ZERO (mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0)))
            #define EPSILON 0.1

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
            extern float kernel(vec4 r, float h, float k);
            extern vec4 grad_w(vec4 r, float h, float k);

            uniform int dim, mode;
            uniform float norm_const, h, g, f;

            float trace(mat4 A) {
                float t = A[0][0];
                for(uint i=1; i<dim; i++) t+=A[i][i];
                return t;
            }

            void qr(in mat4 A, out mat4 Q, out mat4 R) {
                Q = I;  R = A;
                for(uint i=0; i<dim; i++) {
                    //get the first column of the (n-i)-square submatrix
                    vec4 a = R[i];
                    for(uint j=0; j<i; j++) { a[j] = 0.0; }

                    //construct the axis to reflect about
                    float alpha = length(a) * sign(a[i]);
                    vec4 ei = vec4(0.0,0.0,0.0,0.0);
                    ei[i] = alpha;

                    //make a hausdorf reflection matrix to reflect a to a diagonal element
                    vec4 v = a - ei;
                    float l2 = dot(v,v);
                    mat4 Qi = l2==0 ? I : I - (2/l2)*outerProduct(v,v);

                    //compose with the QR decomposition
                    Q = Q * transpose(Qi);
                    R = Qi * R;
                }

            }

            mat4 mat_log(in mat4 A) {
                mat4 sum = mat4(0);
                mat4 term = -I;

                const uint N = 5;
                for(uint i = 1; i<=N; i++) {
                    term *= -1 * (A - I);
                    sum += term / float(i);
                }

                return sum;
            }

            //denman-beavers algorithm
            void mat_sqrt(mat4 X, out mat4 sqrt, out mat4 inv_sqrt) {
                mat4 Yi = X;
                mat4 Zi = I;

                const uint N = 15;
                for(uint i=0; i<N; i++){
                    mat4 Y = 0.5*(Yi + inverse(Zi));
                    mat4 Z = 0.5*(Zi + inverse(Yi));
                    Yi = Y;
                    Zi = Z;
                }

                sqrt = Yi;
                inv_sqrt = Zi;
            }

            float mat_dot(mat4 x, mat4 y) {
                return dot(x[0], y[0]) + dot(x[1], y[1]) + dot(x[2], y[2]) + dot(x[3], y[3]);
            }

            float state(float density, float temperature, float c, float d0){
                return (c*c*d0/7)*(pow(density/d0,7)-1);
            }

            float ideal_gas(float density, float temperature, float c, float d0){
                return c*c*(density - d0);
            }

            mat4 hooke(mat4 strain, float normal, float shear) {
                return normal*trace(strain)*I + 2*shear*strain;
            }

            float johnson_cook(float strain, float strain_rate, float A, float B, float C, float N) {
                return (A + B * pow(strain, N)) * (1 + C * log(strain_rate));
            }

            float pressure(uint eq, float density, float temperature, float c, float d0) {
                if(eq==0) {
                    return 0;
                } else if(eq==1) {
                    return state(density, temperature, c, d0);
                } else {
                    return ideal_gas(density, temperature, c, d0);
                }
            }

            mat4 seth_hill(mat4 def, int two_m) {
                mat4 C = transpose(def)*def;
                for(uint i=dim; i<4; i++) C[i][i] = 1.0;

                if(two_m==0) {
                    return 0.5 * mat_log(C);
                } else {
                    uint m = abs(two_m) >> 1;
                    mat4 C_m = I;
                    for(uint i=0; i<m; i++) C_m *= C;
                    if((abs(two_m)&1) == 1) {
                        mat4 s,inv;
                        mat_sqrt(C, s, inv);
                        C_m *= s;
                    }
                    if(two_m < 0) {
                        C_m = inverse(C_m);
                    }
                    return (1.0/float(two_m)) * (C_m - I);
                }

            }

            public mat4 strain_measure(mat4 def) {
                return seth_hill(def, 2);
                // return 0.5*(def + transpose(def));
            }

            public mat4 strain_rate(mat4 def, mat4 def_rate) {
                mat4 d = transpose(def_rate) * def;
                return 0.5 * (d + transpose(d));
            }

            float eq_plastic_strain_sq(mat4 strain) {
                mat4 dev = strain - trace(strain) * I / dim;
                return ((dim-1)/dim) * mat_dot(dev, dev);
            }

            mat4 pk_stress_unrotated(mat4 cauchy, mat4 def_grad, out mat4 Q) {
                //decompose the gradient into a rotational and shear-stretch component
                mat4 R;
                qr(def_grad, Q, R);

                //fast determinant using the rectangular part
                float J = 1;
                for(uint i=0; i<dim; i++) J *= R[i][i];

                mat4 invR = def_grad;
                for(uint i=dim; i<4; i++) invR[i][i] = 1.0;
                invR = inverse(invR);
                for(uint i=dim; i<4; i++) invR[i][i] = 0.0;

                return J* Q*transpose(cauchy)*transpose(Q)*transpose(invR);
            }

            mat4 pk_stress_unrotated(mat4 cauchy, mat4 def_grad) {
                mat4 Q;
                return pk_stress_unrotated(cauchy, def_grad, Q);
            }

            mat4 cauchy_to_pk(mat4 cauchy, mat4 def_grad) {
                for(uint i=dim; i<4; i++) def_grad[i][i] = 1.0;
                return determinant(def_grad) * inverse(def_grad) * cauchy;
            }

            mat4 cauchy_to_nominal(mat4 cauchy, mat4 def_grad) {
                return transpose(cauchy_to_pk(cauchy, def_grad));
            }

            layout(std430) buffer particle_list { readonly restrict Particle particles[]; };
            layout(std430) buffer sold_particle_list { readonly restrict SolidParticle solids[]; };
            layout(std430) buffer boundary_list { readonly restrict Particle boundary[]; };
            layout(std430) buffer derivatives { restrict Particle forces[]; };
            layout(std430) buffer solid_derivatives { restrict SolidParticle stresses[]; };

            layout(std430) buffer material_list { readonly restrict Material materials[]; };
            layout(std430) buffer strain_list { readonly restrict mat4 strains[][3]; };

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
                bool elastic = s_id!=0xFFFFFFFF && (materials[mat_id].normal_stiffness!=0 || materials[mat_id].shear_stiffness!=0);

                //local variables for storing the result
                float den = 0;
                vec4 force = vec4(0,0,0,0);

                //ids for splitting the for loops
                uint sublocal_id = gl_LocalInvocationID.x/3;
                uint num_sublocal = gl_WorkGroupSize.x/3;

                //get this particle's reference position bucket and offset depending on our local id
                ivec3 b_pos = local_bucket_pos(indices[p_id].ref_index);

                mat4 pk2_stress = ZERO;

                //elastic forces
                if(elastic) {

                    float lambda = materials[mat_id].normal_stiffness;
                    float mu = materials[mat_id].shear_stiffness;
                    float norm_damp = materials[mat_id].normal_damp;
                    float shear_damp = materials[mat_id].shear_damp;

                    mat4 def = strains[s_id][1];
                    mat4 def_rate = strains[s_id][2];
                    // mat4 strain = particles[id].stress;
                    mat4 strain = strain_measure(def) + solids[s_id].stress;
                    mat4 rate_of_strain = strain_rate(def, def_rate);
                    pk2_stress = hooke(strain, lambda, mu) + hooke(rate_of_strain, norm_damp, shear_damp);
                    mat4 stress = pk2_stress;
                    stress = def * transpose(stress);

                    if(in_bounds(b_pos, subdivisions)) {
                        mat4 correction = strains[s_id][0];
                        float V1 = m1 / d1;

                        uint bucket_id = bucket_index(uvec3(b_pos), subdivisions);
                        uint start = buckets[bucket_id].count[0];
                        uint ref_count = buckets[bucket_id].count[1];
                        for(uint j=start+sublocal_id; j<start+ref_count; j+=num_sublocal){
                            uint p_id2 = particle_index(bucket_id, j);
                            uint s_id2 = particles[p_id2].solid_id;
                            if(particles[p_id2].mat != mat_id) continue;

                            float V2 = m1 / particles[p_id2].den;

                            mat4 def2 = strains[s_id2][1];
                            mat4 def_rate2 = strains[s_id2][2];
                            // mat4 strain2 = particles[id2].stress;
                            mat4 strain2 = strain_measure(def2) + solids[s_id2].stress;
                            mat4 strain_rate2 = strain_rate(def2, def_rate2);
                            mat4 stress2 = hooke(strain2, lambda, mu) + hooke(strain_rate2, norm_damp, shear_damp);
                            stress2 = def2 * transpose(stress2);

                            vec4 r = solids[s_id2].ref_pos - solids[s_id].ref_pos;

                            vec4 el_force = -V1 * V2 * (stress * correction + stress2 * strains[s_id2][0]) * grad_w(r,h,norm_const);
                            // vec4 el_force = V2 * stress2 * strains[id2][0] * grad_w(r,h,norm_const);
                            for(uint k=dim; k<4; k++) el_force[k] = 0;

                            force += el_force;
                        }
                    }

                }

                //get this particle's bucket and offset to a neighboring bucket depending on our local id
                b_pos = local_bucket_pos(indices[p_id].pos_index);

                //fluid forces
                if(in_bounds(b_pos, subdivisions)) {

                    //save these properties for convenience
                    uint state_eq = materials[mat_id].state_eq;
                    float c1 = materials[mat_id].sound_speed;
                    float f1 = materials[mat_id].visc;
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
                        float c2 = materials[mat_2].sound_speed;

                        vec4 r,v;
                        float d2;

                        if(j<bc) {
                            r = boundary[p_id2].pos - particles[p_id].pos;
                            v = boundary[p_id2].vel - particles[p_id].vel;
                            d2 = boundary[p_id2].den + d1;
                            c2 = 5000;
                            state_eq2 = state_eq==0 ? 1 : state_eq;
                        } else {
                            r = particles[p_id2].pos - particles[p_id].pos;
                            v = particles[p_id2].vel - particles[p_id].vel;
                            d2 = particles[p_id2].den;
                        }
                        float p2 = pressure(state_eq2, d2, 0, c2, materials[mat_2].target_den);

                        //density update
                        if(!elastic || mat_id==mat_2) {
                            den += m2 * dot(v, grad_w(r, h, norm_const));
                        }

                        vec4 contact_force = vec4(0,0,0,0);
                        bool elastic2 = materials[mat_2].normal_stiffness!=0 || materials[mat_2].shear_stiffness!=0;

                        //hourglass restoring force and contact forces
                        if(elastic || elastic2) {
                            vec4 dx = r;
                            vec4 dX = solids[s_id2].ref_pos - solids[s_id].ref_pos;
                            float contact = h;
                            float r_cut = 2*contact;

                            bool in_contact = dot(dX,dX)>(r_cut*r_cut) && dot(dx,dx)<r_cut*r_cut;

                            if(mat_id == mat_2 || in_contact) {
                                float l = length(dx);
                                if(mat_id == mat_2) {
                                    mat4 F1 = strains[s_id][1];
                                    mat4 F2 = strains[s_id2][1];

                                    vec4 err = 0.5*(F1 + F2)*dX - dx;

                                    force -= (100000000*m1*m2/(d1*d2)) *
                                            (0.5*dot(err,dx)/(l+10*EPSILON*h)) *
                                            (1/(l*l+EPSILON*h*h)) *
                                            kernel(dX, h, norm_const) *
                                            dx/(l+EPSILON*h);
                                }

                                if(in_contact) {
                                    float r_geom = contact*contact / r_cut;
                                    float dr = max(r_cut - l, 0);
                                    contact_force = -5000*sqrt(dr*r_geom) * r / l;
                                    force += contact_force;
                                }
                            }

                        }

                        //pressure force
                        if(!elastic || !elastic2) {
                            vec4 pressure_force = m1*m2*(
                                p1/(d1*d1) +
                                p2/(d2*d2)
                            ) * grad_w(r, h, norm_const);
                            force += pressure_force;
                            if(mat_id != mat_2) contact_force += pressure_force;
                        }

                        //viscocity
                        if(!elastic) force += m1*m2*(f1+f2)*dot(r,grad_w(r, h, norm_const))*v/(d1*d2*(dot(r,r)+EPSILON*h*h));

                        //friction
                        if((elastic || elastic2 || j<bc) && mat_id != mat_2) {
                            vec4 r_inv = r / dot(r,r);
                            vec4 normal_force = r_inv * dot(contact_force, r);
                            vec4 tangent_vel = v - r_inv* dot(v, r);
                            force += (f1 + f2) * length(normal_force) * normalize(tangent_vel);
                        }

                        //artificial viscocity
                        if(j>=bc) force -= m1*m2*(
                            (f*h*(c1+c2))*dot(v, r) / ((d1+d2)*(dot(r,r)+EPSILON*h*h))
                        )*grad_w(r, h, norm_const);

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
                    memoryBarrierShared();
                }

                if(gid==0) {

                    forces[p_id].mat = mat_id;
                    forces[p_id].den = 0;
                    forces[p_id].vel = vec4(0,0,0,0);
                    forces[p_id].pos = particles[p_id].vel;
                    stresses[s_id].ref_pos = vec4(0,0,0,0);
                    stresses[s_id].stress = ZERO;

                    //add up each local unit's contributions
                    for(uint i=0; i<shadow; i++) {
                        forces[p_id].den += shared_den[i];
                        forces[p_id].vel += shared_force[i];
                    }

                    //get the strain-rate
                    if(elastic) {
                        // mat4 def = strains[id][1];
                        // mat4 def_inv = def;
                        // for(uint i=dim; i<4; i++) def_inv[i][i] = 1.0;
                        // float J = determinant(def_inv);
                        // def_inv = inverse(def_inv);
                        //
                        // mat4 def_rate = strains[id][2];
                        // for(uint i=dim; i<4; i++) def_rate[i][i] = 1;
                        // forces[id].den += trace(-def_rate*def_inv/J) * materials[mat_id].start_den;

                        // mat4 Q, R;
                        // qr(strains[id][1], Q, R);
                        //
                        // mat4 K = def_rate * def_inv;
                        // mat4 D = 0.5 * (K + transpose(K));
                        // mat4 d = transpose(Q) * D * Q;

                        // mat4 d = transpose(def_rate) * def;
                        // d = 0.5 * (d + transpose(d));

                        // mat4 C = transpose(def) * def;
                        // mat4 d = transpose(def_rate) * def;
                        // d = 0.5 * (d + transpose(d));
                        // d = (d*C*C + C*d*C + C*C*d)/3;

                        // forces[id].stress = hooke(d, materials[mat_id].normal_stiffness, materials[mat_id].shear_stiffness);
                        // forces[id].stress = d;

                        if(materials[mat_id].plastic) {

                            float A = materials[mat_id].yield_strength;
                            float B = materials[mat_id].work_hardening;
                            float C = materials[mat_id].kinematic_hardening;
                            float n = materials[mat_id].work_hardening_exp;
                            // float m = materials[mat_id].thermal_softening;
                            float T = materials[mat_id].relaxation_time;

                            float eq_strain = eq_plastic_strain_sq(solids[s_id].stress);
                            float stress_norm = mat_dot(pk2_stress, pk2_stress);
                            // float yield_stress = johnson_cook(eq_strain, 1, A, B, C, n/2);
                            float yield_stress = johnson_cook(eq_strain, 1, A, B, C, n/2);
                            if(yield_stress*yield_stress <= stress_norm) {
                                stress_norm = sqrt(stress_norm);
                                // forces[id].stress = pk2_stress * (1 - yield_stress/stress_norm) / T;
                                stresses[s_id].stress = (stress_norm - yield_stress)/T * (pk2_stress / stress_norm);
                                // forces[id].stress = ZERO;
                            }
                        }

                    }

                    //get acceleration from force and gravity
                    forces[p_id].vel /= particles[p_id].den;
                    forces[p_id].vel += vec4(0,-g,0,0);


                }

            }
    }

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
                        memoryBarrierShared();
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
                        memoryBarrierShared();
                    }

                    if(gid==0) {
                        strains[s_id][0] = correction[0];
                        strains[s_id][1] = deformation[0];
                        strains[s_id][2] = def_rate[0];
                    }

                }

            }


    }



    pub fn compute_forces(
        force: &mut compute_force::Program,
        strain: &mut compute_strain::Program,
        buckets: &mut NeighborList,
        particles: ParticleState
    ) -> ParticleState {

        #[allow(mutable_transmutes)]
        particles.map(
            |p| unsafe {
                let prof = crate::PROFILER.as_mut().unwrap();
                prof.new_segment("Bucketting".to_owned());

                let (indices, buckets) = buckets.update_contents(&p);


                let mut dest = p.mirror();
                let particles = p.particles();
                let solids = p.solids();
                let materials = p.materials();

                //NOTE: we know for CERTAIN that neither of these are modified by the shader,
                //so against all warnings, we are going to transmute them to mutable

                let ub_mat: &mut Materials = ::std::mem::transmute::<&Materials,&mut Materials>(materials);
                let ub = ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(&*particles);
                let ub_s = ::std::mem::transmute::<&SolidParticleBuffer,&mut SolidParticleBuffer>(&*solids);
                let ub_bound = ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(p.boundary());

                prof.new_segment("Forces".to_owned());

                let mut strains = Buffer::<[[mat4;3]],Read>::uninitialized(&particles.gl_provider(), solids.len());
                strain.compute(strains.len() as u32, 1, 1, ub, ub_s, ub_mat, indices, &mut strains, buckets);

                // gl::Finish();
                //
                // let mut i = 0;
                // for m in strains.read_into_box().into_iter() {
                //     // println!("{}: {:?}", i, m[0].value);
                //     // println!("{}: {:?}", i, m[1].value);
                //     println!("{}: {:?}", i, m[2].value);
                //     i += 1;
                // }
                // println!();
                //
                // gl::Finish();

                let (mut dest_p, mut dest_s) = dest.all_particles_mut();

                force.compute(
                    p.particles().len() as u32, 1, 1,
                    ub, ub_s, ub_bound, dest_p, dest_s,
                    ub_mat, &mut strains,
                    indices, buckets
                );


                prof.end_segment();

                dest
            }
        )
    }

}

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
    force: RefCell<compute_force::Program>,
    strain: RefCell<compute_strain::Program>

}

impl FluidSim {

    pub fn with_integrator<I: VelIntegrator+Sized+'static>(
        gl: &GLProvider,
        fluids: &[MaterialRegion],
        bounds: AABB, kernel_rad: f32,
        integrator: I, timestep: f32, subticks: uint,
        gravity: f32, artificial_viscocity: f32,
    ) -> Result<Self, GLError> {
        Self::new(gl, fluids, bounds, kernel_rad, Box::new(integrator), timestep, subticks, gravity, artificial_viscocity)
    }

    pub fn new(
        gl: &GLProvider,
        fluids: &[MaterialRegion],
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
            particles: Particles::new(gl, materials.into_boxed_slice(), particles.into_boxed_slice(), boundary.into_boxed_slice()),
            state: Vec::new().into_boxed_slice(),

            neighbor_list: RefCell::new(NeighborList::new(gl, bounds, kernel_rad)),

            force: RefCell::new(compute_force::init(gl).unwrap()),
            strain: RefCell::new(compute_strain::init(gl).unwrap()),
        };

        use self::kernel::norm_const;
        let mut force = fs.force.borrow_mut();
        let mut strain = fs.strain.borrow_mut();

        *force.dim = dim as i32;
        *force.norm_const = norm_const(dim, kernel_rad);
        *force.h = kernel_rad;
        *force.f = artificial_viscocity;
        *force.g = gravity;

        *strain.dim = dim as u32;
        *strain.h = kernel_rad;
        *strain.norm_const = norm_const(dim, kernel_rad);

        drop(force);
        drop(strain);

        Ok(fs)
    }

    pub fn kernel_radius(&self) -> f32 { *self.force.borrow().h }

    pub fn time(&self) -> f32 {self.time}

    pub fn particles(&self) -> &Particles {
        &self.particles
    }

    pub fn add_particles(&mut self, obj: MaterialRegion, offset: Option<vec4>) {
        let (mut p, mat) = obj.gen_particles(*self.force.borrow().h, 0);

        if let Some(t) = offset {
            for x in p.iter_mut() {
                x.pos[0] += t[0];
                x.pos[1] += t[1];
                x.pos[2] += t[2];
                x.pos[3] += t[3];
            }
        }

        self.state[0].map_mut(|particles| particles.add_particles(mat, p.into_boxed_slice()));
        self.particles = self.state[0].clone().map_into(|p| p).unwrap();

    }

}

impl ::ar_engine::engine::Component for FluidSim {

    #[inline]
    fn init(&mut self) {
        let neighbors = &self.neighbor_list;
        let forces = &self.force;
        let strains = &self.strain;

        self.state = self.integrator.init_with_vel(
            ParticleState::new(self.particles.clone()),
            self.timestep / self.subticks as f32,
            & |_t, state| state.velocity(),
            & |_t, state| compute_forces(
                &mut forces.borrow_mut(),
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
        for _ in 0..self.subticks {
            self.time += dt;

            let neighbors = &self.neighbor_list;
            let forces = &self.force;
            let strains = &self.strain;

            self.particles = self.integrator.step_with_vel(
                self.time,
                self.state.as_mut(),
                dt,
                & |_t, state| state.velocity(),
                & |_t, state| compute_forces(
                    &mut forces.borrow_mut(),
                    &mut strains.borrow_mut(),
                    &mut neighbors.borrow_mut(),
                    state
                )
            ).map_into(|p| p).unwrap();

        }

        prof.new_segment("Graphics".to_owned());

    }
}
