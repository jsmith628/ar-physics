#![feature(maybe_uninit_ref)]
#![recursion_limit="2048"]

#[macro_use]
extern crate gl_struct;
extern crate glfw;
extern crate toml;
extern crate stl_io;

extern crate ar_physics;
extern crate ar_engine;
extern crate numerical_integration;


use std::sync::Arc;
use std::cell::RefCell;

use gl_struct::*;
use gl::types::*;
use gl_struct::glsl_type::{vec4, vec3};

use toml::Value;

use stl_io::*;

use std::io::*;
use std::mem::*;
use std::io::Read;
use std::fs::File;
use std::collections::HashMap;

use ar_physics::soft_body::*;
use ar_physics::soft_body::material_region::*;
use ar_physics::soft_body::particle_state::*;

use ar_engine::engine::{Engine};
use ar_engine::timer::{ConstantTimer};

use numerical_integration::*;

pub use self::test_mods::*;
pub use self::test_mods::Surface;

mod test_mods;

glsl!{$

    pub mod ParticleShader {
        @Rust use ar_physics::soft_body::kernel::{normalization_constant, kernel};

        @Vertex
            #version 460

            public struct ColorGradient {
                vec4 c1,c2,c3;
            };

            uniform float densities[32];
            uniform vec4 c1[32];
            uniform vec4 c2[32];
            uniform vec4 c3[32];

            uniform mat4 trans;

            in float den;
            in uint mat;
            in vec4 ref_pos;
            in vec4 pos;
            in vec4 vel;
            // in mat4 strain;

            out vec4 frag_color;
            out vec3 part_pos;
            out vec3 look_basis[3];
            out float depth;

            vec3 hsv2rgb(vec3 c) {
                vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
                return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
            }

            void main() {

                float d0 = densities[mat];

                float d = (den-d0)/(d0);
                if(d<0){
                    frag_color = mix(c2[mat], c1[mat], -d/5);
                }else{
                    frag_color = mix(c2[mat], c3[mat], d/5);
                }

                frag_color.a = min(1.0, frag_color.a);

                part_pos = pos.xyz;

                // mat4 inv = inverse(trans);
                mat4 inv = mat4(vec4(1,0,0,0),vec4(0,1,0,0),vec4(0,0,1,0),vec4(0,0,0,1));
                for(uint i=0; i<3; i++){
                    vec4 basis = vec4(0,0,0,0);
                    basis[i] = i<2 ? 1 : -1;
                    look_basis[i] = (inv * basis).xyz;
                }

                vec3 pos2 = (trans*(vec4(pos.xyz, 1)-vec4(0,0,0.0,0))).xyz+vec3(0,0,0.0);
                part_pos = pos2.xyz;
                gl_Position = vec4(pos2,pos2.z+1);
                depth = pos2.z+1;
            }

        @Fragment
            #version 460

            extern float normalization_constant(int n, float h);
            extern float kernel(vec4 r, float h, float k);

            uniform float render_h;

            uniform bool lighting;
            uniform vec3 light;
            uniform vec3 light_color;
            uniform float ambient_brightness;
            uniform float diffuse_brightness;
            uniform float specular_brightness;

            in vec4 frag_color;
            in vec3 part_pos;
            in vec3 look_basis[3];
            in float depth;

            layout(location = 0, index = 0) out vec4 output_color;

            void main() {
                const float s = 2;

                vec2 p = (gl_PointCoord - vec2(0.5,0.5))*2;
                p.y *= -1;
                vec2 d = p * 2 * depth;
                // vec2 d = p;
                if(frag_color.a==0 || dot(d,d)>1.0){
                    discard;
                } else {
                    if(lighting) {
                        vec3 normal = d.x * look_basis[0] + d.y * look_basis[1];
                        normal += sqrt(1.0 - normal.x*normal.x - normal.y*normal.y) * look_basis[2];
                        vec3 surface_point = part_pos + (render_h)*(normal);

                        vec3 light_vec = light - surface_point;
                        float l = inversesqrt(dot(light_vec,light_vec));
                        light_vec = light_vec * (l*l*l);

                        float diffuse = dot(light_vec, normal);
                        float spec = max(20*dot(-look_basis[2], reflect(light_vec, normal)),0);
                        spec *= spec * spec;

                        output_color.rgb = frag_color.rgb*ambient_brightness;
                        output_color.rgb += diffuse*diffuse_brightness*light_color;
                        // gl_FragColor.rgb = mix(
                        //     ,
                        //     diffuse_color,
                        //     max(diffuse*diffuse_brightness,0)
                        // );
                        output_color.rgb += light_color * spec * specular_brightness;
                        output_color.a = 1.0;
                    } else {
                        float k = normalization_constant(3, 3);
                        output_color = vec4(
                            frag_color.rgb,
                            frag_color.a * kernel(vec4(d,0,0), 1.0, k)/kernel(vec4(0,0,0,0), 1.0, k)
                        );
                    }
                }
            }

    }

}

fn to_vec4(v: &Vec<Value>) -> vec4 {
    let mut p = vec4::default();
    for i in 0..4 { if i<v.len() {p[i] = v[i].as_float().unwrap() as f32;} }
    return p;
}

fn as_bool_or(val: &Value, name: &str, def: bool) -> bool {
    val.as_table().unwrap().get(name).map(|v| v.as_bool().unwrap()).unwrap_or(def)
}

fn as_float_or(val: &Value, name: &str, def: f64) -> f64 {
    val.as_table().unwrap().get(name).map(|v| v.as_float().unwrap()).unwrap_or(def)
}

fn as_int_or(val: &Value, name: &str, def: i64) -> i64 {
    val.as_table().unwrap().get(name).map(|v| v.as_integer().unwrap()).unwrap_or(def)
}

fn as_str_or<'a>(val: &'a Value, name: &str, def: &'a str) -> &'a str {
    val.as_table().unwrap().get(name).map(|v| v.as_str().unwrap()).unwrap_or(def)
}

// fn as_string_or(val: &Value, name: &str, def: String) -> String {
//     val.as_table().unwrap().get(name).map(|v| v.as_str().unwrap().to_owned()).unwrap_or(def)
// }

fn as_float_vec_or(val: &Value, name: &str, def: Vec<f64>) -> Vec<f64> {
    val.as_table().unwrap().get(name).map(
        |v| v.as_array().unwrap().iter().map(|f| f.as_float().unwrap()).collect()
    ).unwrap_or(def)
}

fn as_vec4_or(val: &Value, name: &str, def: vec4) -> vec4 {
    val.as_table().unwrap().get(name).map(|v| to_vec4(&v.as_array().unwrap())).unwrap_or(def)
}

fn as_vec3_or(val: &Value, name: &str, def: vec3) -> vec3 {
    val.as_table().unwrap().get(name).map(
        |v| {
            let v4 = to_vec4(&v.as_array().unwrap());
            [v4[0],v4[1],v4[2]].into()
        }
    ).unwrap_or(def)
}

fn main() {

    //parse cli arguments
    enum ParserState { Default, Width, Height, RecordWidth, RecordHeight, FPS, TPS }

    let mut config_loc = None;
    let (mut win_w, mut win_h) = (640*3/2, 480*3/2);
    let (mut rec_w, mut rec_h) = (None, None);
    let (mut fps, mut tps) = (75.0, 75.0);
    let mut record = false;

    let mut state = ParserState::Default;

    for arg in std::env::args() {

        let mut reset = true;

        match state {
            ParserState::Width => win_w = arg.parse::<u32>().unwrap(),
            ParserState::Height => win_h = arg.parse::<u32>().unwrap(),
            ParserState::RecordWidth => rec_w = arg.parse::<u32>().ok(),
            ParserState::RecordHeight => rec_h = arg.parse::<u32>().ok(),
            ParserState::FPS => fps = arg.parse::<f64>().unwrap(),
            ParserState::TPS => tps = arg.parse::<f64>().unwrap(),

            ParserState::Default => {
                reset = false;
                match arg.as_str() {
                    "-w"|"--width" => state = ParserState::Width,
                    "-h"|"--height" => state = ParserState::Height,
                    "-rw"|"--record-width" => state = ParserState::RecordWidth,
                    "-rh"|"--record-height" => state = ParserState::RecordHeight,
                    "-f"|"--fps" => state = ParserState::FPS,
                    "-t"|"--tps" => state = ParserState::TPS,
                    "-r"|"--record" => record = true,
                    _ => config_loc = Some(arg)
                }
            }
        }

        if reset {state = ParserState::Default;}

    }

    let rec_w = rec_w.unwrap_or(win_w);
    let rec_h = rec_h.unwrap_or(win_h);

    if let Some(path) = config_loc {

        //
        //Get and parse the configuration
        //

        let config_toml = {

            let file = File::open(path).unwrap();
            let mut reader = BufReader::new(file);
            let mut dest = String::new();

            reader.read_to_string(&mut dest).unwrap();
            dest.parse::<Value>().unwrap()
        };

        let cfg = Config::parse(&config_toml);

        //
        //Set up GL context and window
        //

        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        let mut window = glfw.create_window(win_w, win_h, cfg.title, glfw::WindowMode::Windowed).unwrap().0;
        if record {window.set_resizable(false);}

        glfw::Context::make_current(&mut window);
        glfw.set_swap_interval(glfw::SwapInterval::None);

        let gl_provider = unsafe {
            GLProvider::load(|s| ::std::mem::transmute(glfw.get_proc_address_raw(s)))
        };
        let mut context = Context::init(&gl_provider);
        let mut shader = ParticleShader::init(&gl_provider).unwrap();

        //
        //load shader settings
        //

        *shader.render_h = cfg.render_h;
        *shader.lighting = cfg.lighting.into();
        *shader.light = cfg.light.pos;
        *shader.ambient_brightness = cfg.light.ambient;
        *shader.diffuse_brightness = cfg.light.diffuse;
        *shader.specular_brightness = cfg.light.specular;
        *shader.light_color = cfg.light.color;

        //set the colors and densities
        for (id, obj) in cfg.objects.iter().enumerate() {
            shader.c2[id] = obj.colors[0];
            shader.c1[id] = obj.colors[1];
            shader.c3[id] = obj.colors[2];
            shader.densities[id] = obj.region.mat.start_den;
        }

        //Construct the iterator
        let integrator: Box<dyn VelIntegrates<_, _>> = {
            match cfg.integrator.to_lowercase().as_ref() {
                "verlet" => Box::new(VelocityVerlet),
                "euler" => Box::new(EULER),
                "midpoint" => Box::new(MIDPOINT),
                "rk1" => Box::new(RK1),
                "rk2" => Box::new(RK2),
                "heun2" | "heun" => Box::new(HEUN2),
                "ralston" => Box::new(RALSTON),
                "rk3" => Box::new(RK3),
                "heun3" => Box::new(HEUN3),
                "rk4" => Box::new(RK4),
                "rk_3_8" => Box::new(RK_3_8),
                _ => panic!("Invalid integrator")
            }
        };

        let interactions = cfg.interactions.iter().map(|l| &l[0..]).collect::<Vec<_>>();
        let objects = cfg.objects.clone().into_iter().map(|x| x.region).collect::<Vec<_>>();

        if record {unsafe {ar_physics::LOGGING = false}; }

        let mut engine = Engine::new();
        engine.add_component(
            "world",
            if tps < 0.0 {ConstantTimer::new_uncapped()} else {ConstantTimer::from_tps(tps)},
            FluidSim::new(&gl_provider,
                &objects[0..], &interactions[0..],
                AABB{min:cfg.min, dim:cfg.dim}, cfg.h,
                integrator, cfg.dt, cfg.subticks,
                cfg.g, cfg.alpha
            ).unwrap()
        );

        let world = engine.get_component::<FluidSim>("world").unwrap();

        let window1 = Arc::new(RefCell::new(window));
        let window2 = window1.clone();

        let (mut x, mut y) = window1.borrow().get_cursor_pos();
        let (mut l_pressed, mut m_pressed, mut r_pressed) = (false, false, false);
        let (mut trans_x, mut trans_y) = (-cfg.camera_pos[0],-cfg.camera_pos[1]);

        let mut pixels = if record {
            Some(vec![0u8; 3usize * rec_w as usize * rec_h as usize].into_boxed_slice())
        } else {None};

        let [color,depth] = unsafe {
            let mut rb = MaybeUninit::<[GLuint;2]>::uninit();
            gl::GenRenderbuffers(2, &mut rb.assume_init_mut()[0] as *mut GLuint);
            rb.assume_init()
        };

        let fb = unsafe {
            let mut fb = MaybeUninit::uninit();
            gl::GenFramebuffers(1, fb.assume_init_mut());
            fb.assume_init()
        };

        println!("{} {} {}", fb, color, depth);

        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, fb);

            gl::BindRenderbuffer(gl::RENDERBUFFER, color);
            gl::RenderbufferStorage(gl::RENDERBUFFER, gl::RGBA8, rec_w as GLint, rec_h as GLint);
            gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, color);

            gl::BindRenderbuffer(gl::RENDERBUFFER, depth);
            gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH24_STENCIL8, rec_w as GLint, rec_h as GLint);
            gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_STENCIL_ATTACHMENT, gl::RENDERBUFFER, depth);

            gl::BindRenderbuffer(gl::RENDERBUFFER, 0);

            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            match status {
                gl::FRAMEBUFFER_UNDEFINED => panic!("framebuffer undefined"),
                gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => panic!("framebuffer incomplete attachment"),
                gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => panic!("framebuffer incomplete missing attachment"),
                gl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => panic!("framebuffer incomplete draw buffer"),
                gl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER => panic!("framebuffer incomplete read buffer"),
                gl::FRAMEBUFFER_UNSUPPORTED => panic!("framebuffer unsupported"),
                gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => panic!("framebuffer incomplete multisample"),
                gl::FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS => panic!("framebuffer incomplete layer targets"),
                _ => if status != gl::FRAMEBUFFER_COMPLETE { panic!("framebuffer no compete :("); } else {}
            };



        }


        let mut scale = cfg.scale;
        let mut rot = cfg.rot;
        let mut mat_number = objects.len();
        let on_click = cfg.on_click.clone();

        engine.add_component_from_fn(
            "renderer",
            if fps < 0.0 {ConstantTimer::new_uncapped()} else {ConstantTimer::from_tps(fps)},
            move || {

                if let Some(pixels) = pixels.as_mut() {
                    use std::io::Write;

                    unsafe {

                        gl::BindFramebuffer(gl::READ_FRAMEBUFFER, fb);
                        gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

                        gl::ReadPixels(
                            0,0,
                            rec_w as GLsizei, rec_h as GLsizei,
                            gl::RGB,gl::UNSIGNED_BYTE,
                            &mut pixels[0] as *mut u8 as *mut gl::types::GLvoid
                        );

                    }

                    std::io::stdout().write(&**pixels).unwrap();

                }

                let mut w = world.borrow_mut();

                //note, each thing is a column, not row
                {
                    // let s = cfg.scale as f32;
                    let t = rot as f32;
                    *shader.trans = [
                        [t.cos(),        0.0,           t.sin(),   0.0],
                        [ 0.0,             1.0,         0.0,       0.0],
                        [-t.sin(),         0.0,           t.cos(), 0.0],
                        [trans_x as f32, -trans_y as f32, -scale as f32+1.0,       1.0],
                    ].into();
                }

                let (new_x, new_y) = window1.borrow().get_cursor_pos();
                l_pressed = match window1.borrow().get_mouse_button(glfw::MouseButton::Button1) {
                    glfw::Action::Press => true,
                    glfw::Action::Release => false,
                    _ => l_pressed
                };

                r_pressed = match window1.borrow().get_mouse_button(glfw::MouseButton::Button2) {
                    glfw::Action::Press => true,
                    glfw::Action::Release => false,
                    _ => r_pressed
                };

                let m = m_pressed;
                m_pressed = match window1.borrow().get_mouse_button(glfw::MouseButton::Button3) {
                    glfw::Action::Press => true,
                    glfw::Action::Release => false,
                    _ => m_pressed
                };

                if l_pressed {rot += (new_x - x)*0.01;}
                // if m_pressed { scale += (new_x - x)*0.01;}
                if r_pressed {
                    trans_x += (new_x - x)*0.005;
                    trans_y += (new_y - y)*0.005;
                }
                x = new_x;
                y = new_y;

                let (size_x, size_y) = window1.borrow().get_size();
                let min_size = size_x.min(size_y);

                if !m && m_pressed {
                    for obj in on_click.iter() {

                        let (region, relative, colors) = (&obj.region, obj.relative, obj.colors);

                        shader.densities[mat_number] = region.mat.target_den;
                        shader.c1[mat_number] = colors[0];
                        shader.c2[mat_number] = colors[1];
                        shader.c3[mat_number] = colors[2];

                        let offset = match relative {
                            true => Some([
                                2.0*((x as f32 - size_x as f32 /2.0) / min_size as f32),
                                2.0*((size_y as f32/2.0 - y as f32) / min_size as f32),0.0,0.0
                            ].into()),
                            false => None
                        };

                        if w.add_particles(region.clone(), offset) {
                            mat_number += 1;
                        }
                    }
                }

                let particles = w.particles().particles();

                let n = w.particles().particles().len();
                let m = w.particles().boundary().len();
                let (den1, mat1,_, pos1, vel1) = Particle::get_attributes(&w.particles().boundary());
                let (den2, mat2,_, pos2, vel2) = Particle::get_attributes(&*particles);

                let (width, height) = window1.borrow().get_framebuffer_size();
                let s = width.min(height);

                if pixels.is_some() {
                    unsafe {
                        const BUFFERS: [GLenum; 1] = [gl::COLOR_ATTACHMENT0];
                        const CLEAR_COLOR: [GLfloat; 4] = [1.0,0.0,1.0,0.0];
                        gl::BindFramebuffer(gl::FRAMEBUFFER, fb);
                        gl::DrawBuffers(1, &BUFFERS[0] as *const GLenum);

                        let s2 = rec_w.min(rec_h);
                        gl::PointSize(s2 as f32 * *shader.render_h);
                        gl::Viewport(((rec_w-s2)/2) as GLint, ((rec_h-s2)/2) as GLint, s2 as GLint, s2 as GLint);

                        gl::ClearBufferfi(gl::DEPTH_STENCIL, 0, 1.0, 0);
                        gl::ClearBufferfv(gl::COLOR, 0, &CLEAR_COLOR[0] as *const GLfloat);

                        gl::ClearColor(1.0,1.0,1.0,0.0);
                        gl::ClearDepth(1.0);
                        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                    }

                    shader.draw(&mut context, DrawMode::Points, m, den1, mat1, pos1, pos1, vel1);
                    shader.draw(&mut context, DrawMode::Points, n, den2, mat2, pos2, pos2, vel2);
                }

                unsafe {
                    gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                    gl::DrawBuffer(gl::BACK);
                    gl::ClearColor(1.0,1.0,1.0,0.0);
                    gl::ClearDepth(1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                    gl::PointSize(s as f32 * *shader.render_h);
                    gl::Viewport((width-s)/2, (height-s)/2, s, s);
                }

                shader.draw(&mut context, DrawMode::Points, m, den1, mat1, pos1, pos1, vel1);
                shader.draw(&mut context, DrawMode::Points, n, den2, mat2, pos2, pos2, vel2);

                glfw::Context::swap_buffers(&mut *window1.borrow_mut());
                glfw.poll_events();

            }
        );
        
        unsafe {
            gl::Viewport(80*2,0,win_h as i32,win_h as i32);
            gl::Disable(gl::CULL_FACE);
            gl::Disable(gl::RASTERIZER_DISCARD);

            gl::Enable(0x8861);
            gl::Enable(gl::BLEND);

            if !cfg.lighting {
                gl::Disable(gl::DEPTH_TEST);
                gl::BlendEquationSeparate(gl::FUNC_ADD, gl::FUNC_ADD);
                gl::BlendFuncSeparate(gl::ONE_MINUS_DST_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::DST_ALPHA, gl::SRC_ALPHA);
            } else {
                gl::Enable(gl::DEPTH_TEST);
                gl::BlendEquationSeparate(gl::FUNC_ADD, gl::FUNC_ADD);
                gl::BlendFuncSeparate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::ONE, gl::ZERO);
            }

        }

        ::std::thread::sleep(::std::time::Duration::from_millis(1000));

        engine.run_while(move |_| !window2.borrow().should_close());
    } else {
        println!("No config provided. Exiting.");
    }



}
