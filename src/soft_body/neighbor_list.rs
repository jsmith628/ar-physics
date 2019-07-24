
use super::*;
use std::rc::*;


glsl! {$

    pub use self::bucket::*;

    mod bucket {
        @Lib
            public struct Bucket {
                uvec4 index;
                uint count[3];
                uint contents[8][32];
            };

            public struct Index {
                uint pos_index, ref_index;
            }

            public uint bucket_index(uvec3 pos, uvec4 sub){
                return pos.x + pos.y * sub.x + pos.z * sub.x * sub.y;
            }

            public bool in_bounds(ivec3 b_pos, uvec4 sub) {
                return all(greaterThanEqual(b_pos, ivec3(0,0,0))) && all(lessThan(b_pos, ivec3(sub.xyz)));
            }

            public uint particle_index(uint bucket, uint i) {
                return buckets[bucket].contents[i>>5][i&0x1F];
            }
    }

    const GRP_SIZE: u32 = 128;

    mod fill_buckets {
        @Rust
            use super::{Bucket, Index, bucket_index};
            use crate::soft_body::particle_state::{Particle,SolidParticle};
            use crate::soft_body::material_region::AABB;
        @Compute
            #version 460

            #define MAX (4*32)
            #define INVALID_INDEX 0xFFFFFFFF

            #define BOUNDARY 0
            #define REFERENCE 1
            #define POSITION 2

            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            extern struct AABB;
            extern struct Particle;
            extern struct SolidParticle;
            extern struct Bucket;
            extern struct Index;

            layout(std430) buffer particle_list { readonly restrict Particle particles[]; };
            layout(std430) buffer solid_particle_list { readonly restrict SolidParticle solids[]; };
            layout(std430) buffer index_list { writeonly restrict Index indices[]; };

            layout(std430) buffer neighbor_list {
                readonly restrict AABB boundary;
                readonly restrict uvec4 subdivisions;
                restrict Bucket buckets[];
            };

            extern uint bucket_index(uvec3 pos, uvec4 sub);

            uniform uint mode = 0;

            void main() {
                uint id = gl_GlobalInvocationID.x;
                if(id >= particles.length()) return;
                uint s_id = particles[id].solid_id;

                if (mode == REFERENCE && s_id==INVALID_INDEX) {
                    indices[id].ref_index = INVALID_INDEX;
                    return;
                }

                vec4 pos = mode==REFERENCE ? solids[s_id].ref_pos : particles[id].pos;

                if(any(isnan(pos)) || any(isinf(pos))) {
                    if(mode==POSITION) {
                        indices[id].pos_index = INVALID_INDEX;
                    } else if(mode==REFERENCE) {
                        indices[id].ref_index = INVALID_INDEX;
                    }
                    return;
                }

                uvec4 index = uvec4(floor(clamp((pos - boundary.min)/boundary.dim, vec4(0,0,0,0), vec4(1,1,1,1)) * subdivisions));
                uint bucket_id = bucket_index(index.xyz, subdivisions);

                uint start = 0;
                for(uint i=0; i < mode; start+=buckets[bucket_id].count[i], i++);

                uint spot = start + atomicAdd(buckets[bucket_id].count[mode], 1);
                if(spot < MAX) {
                    if(mode==POSITION) indices[id].pos_index = bucket_id;
                    if(mode==REFERENCE) indices[id].ref_index = bucket_id;
                    buckets[bucket_id].contents[spot>>5][spot&0x1F] = id;
                }else{
                    if(mode==POSITION) indices[id].pos_index = INVALID_INDEX;
                    if(mode==REFERENCE) indices[id].ref_index = INVALID_INDEX;
                    atomicMin(buckets[bucket_id].count[mode], MAX-start);
                }
            }

    }

    mod reset_buckets {
        @Rust
            use super::Bucket;
            use crate::soft_body::material_region::AABB;

        @Compute
            #version 460
            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            extern struct AABB;
            extern struct Bucket;

            layout(std430) buffer neighbor_list {
                readonly restrict AABB boundary;
                readonly restrict uvec4 subdivisions;
                writeonly restrict Bucket buckets[];
            };

            uniform uint mode = 0;

            void main() {
                uint id = gl_GlobalInvocationID.x;
                if(id >= buckets.length()) return;
                uint count = buckets[id].count.length();
                for(uint i=mode; i<count; i++) {
                    buckets[id].count[i] = 0;
                }
            }
    }

}

pub(crate) type BucketList = Buffer<(AABB, uvec4, [Bucket]), Read>;
pub(crate) type BucketMap = Buffer<[Index], Read>;

pub struct NeighborList {
    //relevant data
    buckets: BucketList,
    indices: Option<BucketMap>,
    boundary: Weak<ParticleBuffer>,
    active_region: (uvec4, uvec4),

    //shaders for doing the updating
    bucket_fill: fill_buckets::Program,
    bucket_reset: reset_buckets::Program,
}

impl NeighborList {

    fn create_neighbor_list(gl: &GLProvider, boundary:AABB, subdivisions:uvec4) -> BucketList {
        unsafe {
            use ::std::alloc::{Layout, alloc_zeroed};
            use ::std::mem::{size_of, align_of, transmute};
            use ::std::slice::{from_raw_parts_mut};

            let len = subdivisions[0] * subdivisions[1] * subdivisions[2] * subdivisions[3];

            let size = size_of::<(AABB, uvec4, [Bucket;1])>() + (len - 1) as usize * size_of::<Bucket>();
            let ptr = alloc_zeroed(
                Layout::from_size_align(size, align_of::<(AABB, uvec4, [Bucket;1])>()).unwrap()
            );

            let sliced = from_raw_parts_mut(ptr, len as usize);

            let mut boxed = Box::from_raw(transmute::<&mut[u8], &mut(AABB, uvec4, [Bucket])>(sliced));
            boxed.0 = boundary;
            boxed.1 = subdivisions;

            for i in 0..subdivisions[0] {
                for j in 0..subdivisions[1] {
                    for k in 0..subdivisions[2] {
                        let bucket = &mut boxed.2[(i + j*subdivisions[0] + k*subdivisions[0]*subdivisions[1]) as usize];
                        bucket.index[0] = i;
                        bucket.index[1] = j;
                        bucket.index[2] = k;
                        bucket.index[3] = 0;
                        bucket.count = Default::default();
                    }
                }
            }
            Buffer::readonly_from(gl, boxed)
        }
    }

    pub fn new(gl: &GLProvider, bounds: AABB, kernel_radius: f32) -> Self {

        let subdivisions = [
            1.0f32.max(bounds.dim[0]/kernel_radius) as uint,
            1.0f32.max(bounds.dim[1]/kernel_radius) as uint,
            1.0f32.max(bounds.dim[2]/kernel_radius) as uint,
            1.0f32.max(bounds.dim[3]/kernel_radius) as uint,
        ].into();

        NeighborList {
            buckets: Self::create_neighbor_list(gl, bounds, subdivisions),
            indices: None,
            active_region: ([0,0,0,0].into(), subdivisions),
            boundary: Weak::new(),

            bucket_fill: fill_buckets::init(gl).unwrap(),
            bucket_reset: reset_buckets::init(gl).unwrap(),
        }
    }

    #[inline] pub fn bucket_count(&self) -> usize { self.buckets.split_tuple().2.len() }
    #[inline] pub fn active_region(&self) -> (uvec4, uvec4) { self.active_region }
    // #[inline] pub fn bucket_list(&mut self) -> &mut BucketList { &mut self.buckets }

    #[inline]
    pub fn update_contents(&mut self, particles: &Particles) -> (&mut BucketMap, &mut BucketList) {

        //reset the array-list counts for each bucket and make sure the indices list is the right size
        let reset = match &self.indices {
            Some(l) => l.len() < particles.particles().len(),
            None => true
        };
        if reset {
            println!("{}",particles.particles().len());
            self.indices = unsafe {
                Some(Buffer::<[_],_>::uninitialized(&GLProvider::get_current().unwrap(), particles.particles().len().max(1)))
            };
        }

        //NOTE: we know for CERTAIN that this buffer wont be modified by the shader,
        //so against all warnings, we are going to transmute them to mutable
        #[allow(mutable_transmutes)]
        let ub: &mut ParticleBuffer = unsafe { ::std::mem::transmute(particles.particles()) };

        #[allow(mutable_transmutes)]
        let ub_s: &mut SolidParticleBuffer = unsafe { ::std::mem::transmute(particles.solids()) };

        #[inline] fn units(p:u32) -> u32 { (p as f32 / GRP_SIZE as f32).ceil() as u32}

        if !self.boundary.upgrade().map_or(false, |b| &*b as *const ParticleBuffer == particles.boundary() as *const ParticleBuffer) {

            #[allow(mutable_transmutes)]
            let ub_boundary = unsafe {
                ::std::mem::transmute::<&ParticleBuffer,&mut ParticleBuffer>(particles.boundary())
            };

            self.boundary = particles.boundary_weak();

            *self.bucket_reset.mode = 0;
            self.bucket_reset.compute(units(self.bucket_count() as GLuint), 1, 1, &mut self.buckets);

            *self.bucket_fill.mode = 0;
            self.bucket_fill.compute(
                units(ub_boundary.len() as GLuint), 1, 1,
                ub_boundary, ub_s, self.indices.as_mut().unwrap(), &mut self.buckets
            );
        }

        if reset {
            *self.bucket_reset.mode = 1;
            self.bucket_reset.compute(units(self.bucket_count() as GLuint), 1, 1, &mut self.buckets);

            *self.bucket_fill.mode = 1;
            self.bucket_fill.compute(
                units(ub.len() as GLuint), 1, 1,
                ub, ub_s, self.indices.as_mut().unwrap(), &mut self.buckets
            );

        } else {
            *self.bucket_reset.mode = 2;
            self.bucket_reset.compute(units(self.bucket_count() as GLuint), 1, 1, &mut self.buckets);
        }

        *self.bucket_fill.mode = 2;
        self.bucket_fill.compute(
            units(ub.len() as GLuint), 1, 1,
            ub, ub_s, self.indices.as_mut().unwrap(), &mut self.buckets
        );

        if reset {
            // println!("{:?}", self.indices.as_ref().unwrap().read_into_box());
            // println!("{:?}", self.buckets.split_tuple().2.read_into_box().into_iter().filter(|b| b.count[1]>0).collect::<Vec<_>>());
        }

        (self.indices.as_mut().unwrap(), &mut self.buckets)

    }


}
