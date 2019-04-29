#![recursion_limit="2048"]

#[macro_use]
extern crate gl_struct;
extern crate glfw;
extern crate toml;

extern crate ar_physics;
extern crate ar_engine;
extern crate numerical_integration;


use std::rc::Rc;
use std::cell::RefCell;
use gl_struct::*;
use gl_struct::glsl_type::vec4;
use toml::Value;

use ar_physics::soft_body::*;
use ar_physics::soft_body::material_region::*;
use ar_physics::soft_body::particle_state::*;

use ar_engine::engine::{Engine};
use ar_engine::timer::{ConstantTimer};

use numerical_integration::*;


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
            in mat4 strain;

            out vec4 frag_color;

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
                // frag_color = strain * frag_color;
                // frag_color.a = 1;

                // frag_color.xyz = hsv2rgb(vec3(5*atan(ref_pos.y+0.3,ref_pos.x)/(6.28), 1.0, 0.5));

                vec3 pos2 = (trans*(vec4(pos.xyz, 1)-vec4(0,0,0.25,0))).xyz+vec3(0,0,0.25);
                gl_Position = vec4(pos2,pos2.z+1);
            }

        @Fragment
            #version 140

            extern float normalization_constant(int n, float h);
            extern float kernel(vec4 r, float h, float k);

            in vec4 frag_color;

            void main() {
                float k = normalization_constant(3, 3);
                vec2 d = (vec2(0.5,0.5) - gl_PointCoord)*2/gl_FragCoord.w;
                if(frag_color.a==0 || dot(d,d)>1.0){
                    discard;
                } else {
                    gl_FragColor = vec4(
                        frag_color.rgb,
                        frag_color.a * kernel(vec4(d,0,0), 1.0, k)/kernel(vec4(0,0,0,0), 1.0, k)
                    );
                }
            }

    }

}

fn to_vec4(v: &Vec<Value>) -> vec4 {
    let mut p = vec4::default();
    for i in 0..4 { if i<v.len() {p[i] = v[i].as_float().unwrap() as f32;} }
    return p;
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

fn as_string_or(val: &Value, name: &str, def: String) -> String {
    val.as_table().unwrap().get(name).map(|v| v.as_str().unwrap().to_owned()).unwrap_or(def)
}

fn as_vec4_or(val: &Value, name: &str, def: vec4) -> vec4 {
    val.as_table().unwrap().get(name).map(|v| to_vec4(&v.as_array().unwrap())).unwrap_or(def)
}

fn main() {

    //parse cli arguments
    enum ParserState { Default, Width, Height, FPS, TPS }

    let mut config_loc = None;
    let (mut win_w, mut win_h) = (640*3/2, 480*3/2);
    let (mut fps, mut tps) = (75.0, 75.0);

    let mut state = ParserState::Default;

    for arg in std::env::args() {

        let mut reset = true;

        match state {
            ParserState::Width => win_w = arg.parse::<u32>().unwrap(),
            ParserState::Height => win_h = arg.parse::<u32>().unwrap(),
            ParserState::FPS => fps = arg.parse::<f64>().unwrap(),
            ParserState::TPS => tps = arg.parse::<f64>().unwrap(),

            ParserState::Default => {
                reset = false;
                match arg.as_str() {
                    "-w"|"--width" => state = ParserState::Width,
                    "-h"|"--height" => state = ParserState::Height,
                    "-f"|"--fps" => state = ParserState::FPS,
                    "-t"|"--tps" => state = ParserState::TPS,
                    _ => config_loc = Some(arg)
                }
            }
        }

        if reset {state = ParserState::Default;}

    }

    if let Some(path) = config_loc {

        let config = {

            use std::fs::File;
            use std::io::*;

            let file = File::open(path).unwrap();
            let mut reader = BufReader::new(file);
            let mut dest = String::new();

            reader.read_to_string(&mut dest).unwrap();
            dest.parse::<Value>().unwrap()
        };

        let title = as_str_or(&config, "name", "Fluid Test");
        let subticks = as_int_or(&config, "subticks", 1) as u32;
        let h = as_float_or(&config, "kernel_radius", 1.0/64.0) as f32;
        let dt = as_float_or(&config, "time_step", 0.01) as f32;
        let alpha = as_float_or(&config, "artificial_viscocity", 50.0) as f32;
        let g = as_float_or(&config, "gravity", 1.0) as f32;

        let min = as_vec4_or(&config, "min", [-1.0,-1.0,0.0,0.0].into());
        let dim = as_vec4_or(&config, "dim", [2.0,2.0,0.0,0.0].into());

        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        let mut window = glfw.create_window(win_w, win_h, title, glfw::WindowMode::Windowed).unwrap().0;

        glfw::Context::make_current(&mut window);
        window.set_key_polling(true);
        glfw.set_swap_interval(glfw::SwapInterval::None);

        let gl_provider = unsafe {
            GLProvider::load(|s| ::std::mem::transmute(glfw.get_proc_address_raw(s)))
        };
        let mut context = Context::init(&gl_provider);
        let mut shader = ParticleShader::init(&gl_provider).unwrap();

        let fluids = {

            use toml::value::{Table, Array};

            fn parse_region_from_mat(region: &Table) -> Rc<Region> {

                type Reg = Rc<Region>;

                fn fold_array<F:Fn(Reg,Reg) -> Reg>(arr: &Array, f:F) -> Reg {
                    let mut iter = arr.into_iter().map(|s| parse_region(s.as_table().unwrap()));
                    let first = iter.next().unwrap();
                    iter.fold(first, f)
                }

                #[allow(unused_assignments)]
                fn parse_region(region: &Table) -> Reg {

                    let mut boxed = true;
                    let mut min = [0.0,0.0,0.0,0.0].into();
                    let mut dim = [0.0,0.0,0.0,0.0].into();
                    let border = region.get("border").map(|f| f.as_float().unwrap() as f32);

                    if let Some(Value::Array(arr)) = region.get("dim") {
                        boxed = true;
                        dim = to_vec4(arr);
                    } else if let Some(Value::Array(arr)) = region.get("radii") {
                        boxed = false;
                        dim = to_vec4(arr);
                        for i in 0..4 { dim[i] *= 2.0; }
                    } else if let Some(Value::Array(arr)) = region.get("diameters") {
                        boxed = false;
                        dim = to_vec4(arr);
                    } else {
                        //catch possible unions/intersections/differences
                        return parse_region_from_mat(region);
                    }

                    if let Some(Value::Array(arr)) = region.get("min") {
                        min = to_vec4(arr);
                    } else if let Some(Value::Array(arr)) = region.get("center") {
                        min = to_vec4(arr);
                        for i in 0..4 { min[i] -= 0.5*dim[i]; }
                    }

                    if boxed {
                        if let Some(depth) = border {
                            Rc::new(AABB {min: min, dim: dim}.border(depth))
                        } else {
                            Rc::new(AABB {min: min, dim: dim})
                        }
                    } else {
                        for i in 0..4 {
                            dim[i] *= 0.5;
                            min[i] += dim[i];
                        }
                        if let Some(depth) = border {
                            Rc::new(AABE {center: min, radii: dim}.border(depth))
                        } else {
                            Rc::new(AABE {center: min, radii: dim})
                        }
                    }

                }


                if let Some(Value::Array(arr)) = region.get("union") {
                    fold_array(arr, |s1,s2| Rc::new(Union(s1,s2)))
                } else if let Some(Value::Array(arr)) = region.get("difference") {
                    fold_array(arr, |s1,s2| Rc::new(Difference(s1,s2)))
                } else if let Some(Value::Array(arr)) = region.get("intersection") {
                    fold_array(arr, |s1,s2| Rc::new(Intersection(s1,s2)))
                } else if let Some(Value::Table(table)) = region.get("region") {
                    parse_region(table)
                } else {
                    panic!("Material has no table named region or doesn't have union, difference, or intersection arrays");
                }
            }

            fn set_colors_and_den(mat:&Value, shader: &mut ParticleShader::Program, id:usize, den: f32) {
                let color = as_vec4_or(&mat, "color", [0.0,0.0,0.0,1.0].into());
                shader.c2[id] = color;
                shader.c1[id] = as_vec4_or(&mat, "color_low_density", color);
                shader.c3[id] = as_vec4_or(&mat, "color_high_density", color);

                shader.densities[id] = den;
            }

            let mut list = Vec::new();
            let mut mat_number = 0;

            if let Some(Value::Table(boundaries)) = config.get("boundaries") {
                for (_, immobile) in boundaries {
                    let packing = as_float_or(&immobile, "packing", 0.5) as f32;
                    let friction = as_float_or(&immobile, "friction", 0.0) as f32;

                    let mut boundary = MaterialRegion::new_immobile(
                        AABB {min: [0.0,0.0,0.0,0.0].into(), dim: [0.0,0.0,0.0,0.0].into()},
                        packing, friction
                    );

                    boundary.region = parse_region_from_mat(immobile.as_table().unwrap());
                    list.push(boundary);
                    set_colors_and_den(&immobile, &mut shader, mat_number, 1.0);
                    mat_number += 1;
                }
            }

            if let Some(Value::Table(solids)) = config.get("elastics") {
                for (_, solid) in solids {
                    let packing = as_float_or(&solid, "packing", 0.5) as f32;
                    let den = as_float_or(&solid, "density", 1.0) as f32;
                    let b = as_float_or(&solid, "normal_stiffness", 1.0) as f32;
                    let s = as_float_or(&solid, "shear_stiffness", 1.0) as f32;
                    let damp = as_float_or(&solid, "dampening", 0.0) as f32;

                    println!("{} {}", b, s);

                    let mut thing = MaterialRegion::new_elastic(
                        AABB {min: [0.0,0.0,0.0,0.0].into(), dim: [0.0,0.0,0.0,0.0].into()},
                        packing, den, b, s, damp
                    );

                    thing.region = parse_region_from_mat(solid.as_table().unwrap());
                    list.push(thing);
                    set_colors_and_den(&solid, &mut shader, mat_number, den);
                    mat_number += 1;
                }
            }

            if let Some(Value::Table(liquids)) = config.get("liquids") {
                for (_, liquid) in liquids {
                    let packing = as_float_or(&liquid, "packing", 0.5) as f32;
                    let den = as_float_or(&liquid, "density", 1.0) as f32;
                    let c = as_float_or(&liquid, "speed_of_sound", 1.0) as f32;
                    let visc = as_float_or(&liquid, "viscocity", 0.0) as f32;

                    let mut fluid = MaterialRegion::new_liquid(
                        AABB {min: [0.0,0.0,0.0,0.0].into(), dim: [0.0,0.0,0.0,0.0].into()},
                        packing, den, c, visc
                    );

                    fluid.region = parse_region_from_mat(liquid.as_table().unwrap());
                    list.push(fluid);
                    set_colors_and_den(&liquid, &mut shader, mat_number, den);
                    mat_number += 1;
                }
            }

            if let Some(Value::Table(liquids)) = config.get("gasses") {
                for (_, gas) in liquids {
                    let packing = as_float_or(&gas, "packing", 0.5) as f32;
                    let start_den = as_float_or(&gas, "start_density", 1.0) as f32;
                    let tar_den = as_float_or(&gas, "target_density", 0.0) as f32;
                    let c = as_float_or(&gas, "speed_of_sound", 1.0) as f32;
                    let visc = as_float_or(&gas, "viscocity", 0.0) as f32;

                    let mut fluid = MaterialRegion::new_gas(
                        AABB {min: [0.0,0.0,0.0,0.0].into(), dim: [0.0,0.0,0.0,0.0].into()},
                        packing, tar_den, start_den, c, visc
                    );

                    fluid.region = parse_region_from_mat(gas.as_table().unwrap());
                    list.push(fluid);
                    set_colors_and_den(&gas, &mut shader, mat_number, start_den);
                    mat_number += 1;
                }
            }

            list
        };

        let integrator: Box<VelIntegrates<_, _>> = {
            match as_str_or(&config, "integrator", "verlet").to_lowercase().as_ref() {
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

        let mut engine = Engine::new();
        engine.add_component(
            "world",
            if tps < 0.0 {ConstantTimer::new_uncapped()} else {ConstantTimer::from_tps(tps)},
            FluidSim::new(&gl_provider,
                &fluids[0..], AABB{min:min, dim:dim}, h, integrator, dt, subticks, g, alpha
            ).unwrap()
        );

        let world = engine.get_component::<FluidSim>("world").unwrap();

        let window1 = Rc::new(RefCell::new(window));
        let window2 = window1.clone();

        let (mut x, _) = window1.borrow().get_cursor_pos();
        let mut pressed = false;
        let mut rot = 0.0;

        engine.add_component_from_fn(
            "renderer",
            if fps < 0.0 {ConstantTimer::new_uncapped()} else {ConstantTimer::from_tps(fps)},
            move || {

                let (width, height) = window1.borrow().get_framebuffer_size();
                let s = width.min(height);
                unsafe {gl::Viewport((width-s)/2, (height-s)/2, s, s)};

                glfw::Context::swap_buffers(&mut *window1.borrow_mut());
                glfw.poll_events();

                let w = world.borrow();

                let t = rot as f32;
                *shader.trans = [
                    [ t.cos(), 0.0, t.sin(), 0.0],
                    [ 0.0,     1.0, 0.0,     0.0],
                    [-t.sin(), 0.0, t.cos(), 0.0],
                    [ 0.0,     0.0, 0.0,     1.0],
                ].into();



                let (new_x, _) = window1.borrow().get_cursor_pos();
                pressed = match window1.borrow().get_mouse_button(glfw::MouseButton::Button1) {
                    glfw::Action::Press => true,
                    glfw::Action::Release => false,
                    _ => pressed
                };

                if pressed { rot += (new_x - x)*0.01;}
                x = new_x;

                unsafe {
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                    gl::ClearColor(1.0,1.0,1.0,0.0);
                    gl::PointSize(s as f32 * w.kernel_radius());
                }

                // println!("{:?}",
                //     w.particles().buf
                //         .read_into_box()
                //         .into_iter()
                //         .map(|p| p.strain)
                //         .filter(|s| s.value.iter().flatten().fold(false, |b,n| b || *n!=0.0))
                //         .collect::<Vec<_>>()
                // );

                let n = w.particles().buf.len();
                let m = w.particles().boundary.len();
                let (den, mat,ref_pos, pos, vel,strain) = Particle::get_attributes(&w.particles().boundary);
                shader.draw(&mut context, DrawMode::Points, m, den, mat, ref_pos, pos, vel,strain);

                let (den, mat,ref_pos, pos, vel,strain) = Particle::get_attributes(&w.particles().buf);
                shader.draw(&mut context, DrawMode::Points, n, den, mat, ref_pos, pos, vel,strain);

            }
        );

        unsafe {
            gl::Viewport(80*2,0,win_h as i32,win_h as i32);
            gl::Disable(gl::CULL_FACE);
            gl::Disable(gl::DEPTH_TEST);

            gl::Enable(0x8861);
            gl::Enable(gl::BLEND);
            gl::BlendEquationSeparate(gl::FUNC_ADD, gl::FUNC_ADD);
            gl::BlendFuncSeparate(gl::ONE_MINUS_DST_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::DST_ALPHA, gl::SRC_ALPHA);
            // gl::Disable(gl::BLEND);
        }

        ::std::thread::sleep(::std::time::Duration::from_millis(1000));

        engine.run_while(move |_| !window2.borrow().should_close());
    } else {
        println!("No config provided. Exiting.");
    }



}
