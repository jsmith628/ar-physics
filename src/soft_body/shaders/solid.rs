use super::*;

glsl! {$

    pub mod clear_solids {
        @Rust
            use super::*;

            impl Program {
                pub unsafe fn clear_solids(&self, p: &SolidParticleBuffer, dest: &mut SolidParticleBuffer) {
                    #[allow(mutable_transmutes)]
                    let ub = ::std::mem::transmute::<_, &mut SolidParticleBuffer>(p);
                    self.compute(((dest.len() as f32)/128.0).ceil() as GLuint, 1, 1, ub, dest);
                }
            }

        @Compute
            #version 460

            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
            extern struct SolidParticle;

            layout(std430) buffer solids_list { readonly restrict SolidParticle p[]; };
            layout(std430) buffer dest_list { writeonly restrict SolidParticle dest[]; };

            void main() {
                uint id = gl_GlobalInvocationID.x;
                if(id < dest.length()) {
                    dest[id].part_id = p[id].part_id;
                    dest[id].ref_pos = vec4(0,0,0,0);
                    dest[id].stress = mat4(vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0));
                }
            }


    }
}

glsl!{$

    pub mod solid_forces {
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
            #define INVALID_INDEX 0xFFFFFFFF

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

            mat4 hooke(mat4 strain, float normal, float shear) {
                return normal*trace(strain)*I + 2*shear*strain;
            }

            float johnson_cook(float strain, float strain_rate, float A, float B, float C, float N) {
                return (A + B * pow(strain, N)) * (1 + C * log(strain_rate));
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
            layout(std430) buffer derivatives { writeonly restrict Particle forces[]; };
            layout(std430) buffer solid_derivatives { writeonly restrict SolidParticle stresses[]; };

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

                if(gid==0) {
                    forces[p_id].mat = mat_id;
                    forces[p_id].solid_id = s_id;
                    forces[p_id].den = 0;
                    forces[p_id].pos = vec4(0,0,0,0);
                    forces[p_id].vel = vec4(0,0,0,0);
                }

                if(s_id!=INVALID_INDEX){
                    if(gid==0) {
                        stresses[s_id].part_id = p_id;
                        stresses[s_id].ref_pos = vec4(0,0,0,0);
                        stresses[s_id].stress = ZERO;
                    }
                } else {
                    return;
                }


                barrier();

                //save these properties for convenience
                float m1 = materials[mat_id].mass;
                float d1 = particles[p_id].den;
                bool elastic = s_id!=INVALID_INDEX && (materials[mat_id].normal_stiffness!=0 || materials[mat_id].shear_stiffness!=0);

                if(!elastic) { return; }

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
                            if(p_id2==INVALID_INDEX || s_id2==INVALID_INDEX) continue;

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

                    float f1 = materials[mat_id].visc;

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

                        if(p_id2==INVALID_INDEX || s_id2==INVALID_INDEX) continue;

                        float m2 = materials[mat_2].mass;
                        float f2 = materials[mat_2].visc;

                        vec4 r,v;
                        float d2;

                        if(j<bc) {
                            r = boundary[p_id2].pos - particles[p_id].pos;
                            v = boundary[p_id2].vel - particles[p_id].vel;
                            d2 = boundary[p_id2].den + d1;
                        } else {
                            r = particles[p_id2].pos - particles[p_id].pos;
                            v = particles[p_id2].vel - particles[p_id].vel;
                            d2 = particles[p_id2].den;
                        }

                        vec4 contact_force = vec4(0,0,0,0);
                        bool elastic2 = s_id2!=INVALID_INDEX && (materials[mat_2].normal_stiffness!=0 || materials[mat_2].shear_stiffness!=0);

                        //hourglass restoring force and contact forces
                        if(elastic || elastic2) {
                            vec4 dx = r;
                            vec4 dX = solids[s_id2].ref_pos - solids[s_id].ref_pos;
                            float contact = h;
                            float r_cut = 8*contact;

                            bool in_contact = (!elastic || !elastic2 || dot(dX,dX)>(r_cut*r_cut)) && dot(dx,dx)<r_cut*r_cut;

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

                                if(mat_id == mat_2  && in_contact && !materials[mat_id].plastic) {
                                    float r_geom = contact*contact / r_cut;
                                    float dr = max(r_cut - l, 0);
                                    contact_force = -5000*sqrt(dr*r_geom) * r / l;
                                    force += contact_force;
                                }
                            }

                        }

                        //friction
                        if((elastic || elastic2 || j<bc) && mat_id != mat_2) {
                            vec4 r_inv = r / dot(r,r);
                            vec4 normal_force = r_inv * dot(contact_force, r);
                            vec4 tangent_vel = v - r_inv* dot(v, r);
                            force += (f1 + f2) * length(normal_force) * normalize(tangent_vel);
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

                    forces[p_id].den = shared_den[0];
                    forces[p_id].vel = shared_force[0]/particles[p_id].den + vec4(0,-g,0,0);
                    forces[p_id].pos = particles[p_id].vel;

                    //get the strain-rate
                    if(elastic) {

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

                }

            }
    }
}
