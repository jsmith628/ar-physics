#![recursion_limit="1024"]
#![feature(unsized_tuple_coercion)]
#![feature(box_syntax)]
#![feature(trace_macros)]

extern crate ar_physics;
#[macro_use]
extern crate gl_struct;
extern crate ar_engine;
extern crate glfw;

use ar_physics::*;
use gl_struct::*;
use gl_struct::glsl_type::*;

// use std::f32::consts::PI;


glsl!{$

    pub mod ParticleShader {
        @Vertex
            #version 140
            in vec4 pos;
            in vec4 color;
            out vec4 frag_color;
            void main() {
                frag_color = color;
                gl_Position = pos;
            }

        @Fragment
            #version 140
            in vec4 frag_color;
            void main() {
                gl_FragColor = frag_color;
            }

    }

    pub mod ParticleUpdator {

        @Rust
            use ::ar_physics::{Particle, AABB, Bucket, particle_index};

        @Compute

            #version 440

            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            extern struct Particle;
            extern struct AABB;
            extern struct Bucket;

            extern uint particle_index(Bucket b, uint i);

            layout(std430) buffer particles {
                Particle list[];
            };

            layout(std430) buffer neighbor_list {
                readonly restrict AABB boundary;
                readonly restrict uvec4 subdivisions;
                readonly restrict Bucket buckets[];
            };

            void main() {

                Bucket bucket = buckets[gl_GlobalInvocationID.x];

                for(uint i=0; i<bucket.index_count.w; i++) {
                    uint major = i>>5;
                    uint minor = i & 0x1F;

                    const float dt = 0.001;
                    // const float dt = 0.00025;
                    const mat2 rot = mat2(vec2(0,1), vec2(-1,0));
                    vec2 yn = list[particle_index(bucket, i)].pos.xy;

                    vec2 k1 = dt * normalize(rot * (yn));
                    vec2 k2 = dt * normalize(rot * (yn + k1/2.0));
                    vec2 k3 = dt * normalize(rot * (yn + k2/2.0));
                    vec2 k4 = dt * normalize(rot * (yn + k3));


                    // list[particle_index(bucket, i)].pos.xy += k1;
                    // list[particle_index(bucket, i)].pos.xy += k2;
                    list[particle_index(bucket, i)].pos.xy += (k1 + 2*k2 + 2*k3 + k4) / 6;

                    list[particle_index(bucket, i)].vel = vec4(bucket.index_count.xy,0,1) / vec4(subdivisions.xy,1,1);

                }

            }

    }

}


fn main() {

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let mut window = glfw.create_window(640*2, 480*2, "Neighbor Lists", glfw::WindowMode::Windowed).unwrap().0;

    glfw::Context::make_current(&mut window);
    window.set_key_polling(true);
    glfw.set_swap_interval(glfw::SwapInterval::None);

    let gl_provider = unsafe {
        GLProvider::load(|s| ::std::mem::transmute(glfw.get_proc_address_raw(s)))
    };
    let mut context = Context::init(&gl_provider);
    let shader = ParticleShader::init(&gl_provider).unwrap();
    let sorter = fill_buckets::init(&gl_provider).unwrap();
    let eraser = clear_buckets::init(&gl_provider).unwrap();
    let computer = ParticleUpdator::init(&gl_provider).unwrap();

    let num = 600000;
    let mut points = Vec::with_capacity(num);
    for _i in 0..num {
        points.push( Particle {
            pos: [rand::random::<f32>() * 1.0 - 0.5, rand::random::<f32>() * 1.0 - 0.5, 0.0, 1.0].into(),
            vel: [rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()].into()
        });
    }
    let mut particles: Buffer<_,_> = Buffer::immut_from(&gl_provider, points.into_boxed_slice());

    const W: u32 = 512;
    const H: u32 = 512;

    let boundary = AABB {min: [-1.0,-1.0,-1.0,-1.0].into(), dim: [2.0,2.0,2.0,2.0].into()};
    let subdivisions: uvec4 = [W, H, 1,1].into();
    let mut neighbor_list = create_neighbor_list(&gl_provider, boundary, subdivisions);

    unsafe {
        gl::Viewport(80*2,0,480*2,480*2);
        gl::Disable(gl::CULL_FACE);
        gl::Disable(gl::DEPTH_TEST);
        gl::Disable(gl::BLEND);
    }

    let mut tick = 0u64;
    while !window.should_close() {
        let start = ::std::time::Instant::now();

        glfw::Context::swap_buffers(&mut window);
        glfw.poll_events();

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::ClearColor(1.0,1.0,1.0,0.0);
            gl::PointSize(1.0);
        }


        if tick&0 == 0 {
            eraser.compute(particles.len() as u32, 1, 1, &mut neighbor_list);
            sorter.compute(particles.len() as u32, 1, 1, &mut particles, &mut neighbor_list);
        }
        computer.compute(W*H, 1, 1, &mut particles, &mut neighbor_list);

        let (pos, vel) = Particle::get_attributes(&particles);
        shader.draw(&mut context, DrawMode::Points, particles.len(), pos, vel);

        // ::std::thread::sleep(::std::time::Duration::from_millis(300));

        println!("{} {:?}", tick, ::std::time::Instant::now() - start);

        tick+=1;

    }



}
