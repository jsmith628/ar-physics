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
use gl_struct::glsl_type::{vec4, vec3};
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
            #version 140

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

                        gl_FragColor.rgb = frag_color.rgb*ambient_brightness;
                        gl_FragColor.rgb += diffuse*diffuse_brightness*light_color;
                        // gl_FragColor.rgb = mix(
                        //     ,
                        //     diffuse_color,
                        //     max(diffuse*diffuse_brightness,0)
                        // );
                        gl_FragColor.rgb += light_color * spec * specular_brightness;
                        gl_FragColor.a = 1.0;
                    } else {
                        float k = normalization_constant(3, 3);
                        gl_FragColor = vec4(
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

fn as_string_or(val: &Value, name: &str, def: String) -> String {
    val.as_table().unwrap().get(name).map(|v| v.as_str().unwrap().to_owned()).unwrap_or(def)
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
        let lighting = as_bool_or(&config, "lighting", config.as_table().unwrap().get("light_pos").is_some());

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
        glfw.set_swap_interval(glfw::SwapInterval::None);

        let gl_provider = unsafe {
            GLProvider::load(|s| ::std::mem::transmute(glfw.get_proc_address_raw(s)))
        };
        let mut context = Context::init(&gl_provider);
        let mut shader = ParticleShader::init(&gl_provider).unwrap();
        *shader.render_h = h * as_float_or(&config, "particle_render_factor", 1.0) as f32;
        *shader.lighting = lighting.into();
        *shader.light = as_vec3_or(&config, "light_pos", [-3.0,3.0,-3.0].into());
        *shader.ambient_brightness = as_float_or(&config, "ambient_brightness", 1.0) as f32;
        *shader.diffuse_brightness = as_float_or(&config, "diffuse_brightness", 10.0) as f32;
        *shader.specular_brightness = as_float_or(&config, "specular_brightness", 0.0) as f32;
        *shader.light_color = as_vec3_or(&config, "light_color", [1.0,1.0,1.0].into());

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

            fn get_packing(region: &Value, h: f32) -> f32 {
                as_float_or(region, "packing", as_float_or(region, "spacing", 0.5*h as f64)/h as f64) as f32
            }

            if let Some(Value::Table(boundaries)) = config.get("boundaries") {
                for (_, immobile) in boundaries {
                    let packing = get_packing(&immobile, h);
                    let friction = as_float_or(&immobile, "friction", 0.0) as f32;

                    let region = parse_region_from_mat(immobile.as_table().unwrap());
                    list.push(MaterialRegion::new(region, packing, Material::new_immobile(friction)));
                    set_colors_and_den(&immobile, &mut shader, mat_number, 1.0);
                    mat_number += 1;
                }
            }

            if let Some(Value::Table(objects)) = config.get("objects") {
                for (_, obj) in objects {
                    let table = obj.as_table().unwrap();
                    let mut mat = Material::default();
                    mat.start_den = as_float_or(&obj, "start_density", as_float_or(&obj, "density", 1.0)) as f32;
                    mat.target_den = as_float_or(&obj, "target_density", as_float_or(&obj, "density", 1.0)) as f32;
                    mat.sound_speed = as_float_or(&obj, "speed_of_sound", 0.0) as f32;
                    mat.visc = as_float_or(&obj, "viscocity", as_float_or(&obj, "friction", 0.0)) as f32;
                    mat.state_eq = match table.get("state").or(table.get("state_equation")).map(|a| a.as_str().unwrap()) {
                        None => if table.get("speed_of_sound").is_some() {
                            if table.get("start_density").is_some() || table.get("target_density").is_some() {
                                StateEquation::IdealGas
                            } else {
                                StateEquation::Tait
                            }
                        } else{
                            StateEquation::Zero
                        },
                        Some("Tait") | Some("tait") | Some("Liquid") | Some("liquid") => StateEquation::Tait,
                        Some("Ideal_Gas") | Some("ideal_gas") | Some("gas") | Some("Gas") => StateEquation::IdealGas,
                        Some(s) => panic!("Invalid state equation: {}", s)
                    } as u32;

                    mat.normal_stiffness = as_float_or(&obj, "normal_stiffness", 0.0) as f32;
                    mat.shear_stiffness = as_float_or(&obj, "shear_stiffness", 0.0) as f32;
                    mat.normal_damp = as_float_or(&obj, "normal_dampening", 0.0) as f32;
                    mat.shear_damp = as_float_or(&obj, "shear_dampening", 0.0) as f32;

                    if table.get("yield_strength").is_some() || table.get("relaxation_time").is_some() {
                        mat.plastic = true.into();
                        mat.yield_strength = as_float_or(&obj, "yield_strength", 0.0) as f32;
                        mat.work_hardening = as_float_or(&obj, "work_hardening", 0.0) as f32;
                        mat.work_hardening_exp = as_float_or(&obj, "work_hardening_exp", 2.0) as f32;
                        mat.kinematic_hardening = as_float_or(&obj, "kinematic_hardening", 0.0) as f32;
                        mat.thermal_softening = as_float_or(&obj, "thermal_softening", 0.0) as f32;
                        mat.relaxation_time = as_float_or(&obj, "relaxation_time", 1.0) as f32;
                    }

                    let region = parse_region_from_mat(table);

                    let vel = as_vec4_or(&obj, "velocity", [0.0,0.0,0.0,0.0].into());
                    let center = region.bound().center();

                    let mut ang_vel = [0.0;6];
                    if let Some(arr) = table.get("angular_velocity") {
                        let mut i = 0;
                        for val in arr.as_array().unwrap() {
                            ang_vel[i] = val.as_float().unwrap() as f32;
                            i += 1;
                        }
                    }

                    list.push(
                        MaterialRegion::with_vel(
                            region,
                            get_packing(&obj, h),
                            mat,
                            move |p| {
                                let r = [p[0]-center[0], p[1]-center[1], p[2]-center[2], p[3]-center[3]];
                                let w = ang_vel;
                                [
                                    w[1]*r[2] - w[2]*r[1] - w[3]*r[3] + vel[0],
                                    w[2]*r[0] - w[0]*r[2] + w[4]*r[3] + vel[1],
                                    w[0]*r[1] - w[1]*r[0] - w[5]*r[3] + vel[2],
                                    w[3]*r[0] - w[4]*r[1] + w[5]*r[3] + vel[3],
                                ].into()
                            }
                        )
                    );
                    set_colors_and_den(&obj, &mut shader, mat_number, mat.target_den);
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

        let (mut x, mut y) = window1.borrow().get_cursor_pos();
        let (mut l_pressed, mut m_pressed, mut r_pressed) = (false, false, false);
        let (mut trans_x, mut trans_y) = (0.0,0.0);
        let mut scale = 1.0;
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

                //note, each thing is a column, not row
                {
                    // let s = scale as f32;
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

                m_pressed = match window1.borrow().get_mouse_button(glfw::MouseButton::Button3) {
                    glfw::Action::Press => true,
                    glfw::Action::Release => false,
                    _ => m_pressed
                };

                if l_pressed {rot += (new_x - x)*0.01;}
                if m_pressed {scale += (new_x - x)*0.01;}
                if r_pressed {
                    trans_x += (new_x - x)*0.005;
                    trans_y += (new_y - y)*0.005;
                }
                x = new_x;
                y = new_y;

                unsafe {
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                    gl::ClearColor(1.0,1.0,1.0,0.0);
                    gl::PointSize(s as f32 * *shader.render_h * 1.0);
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
                let (den, mat,ref_pos, pos, vel, _) = Particle::get_attributes(&w.particles().boundary);
                shader.draw(&mut context, DrawMode::Points, m, den, mat, ref_pos, pos, vel);

                let (den, mat,ref_pos, pos, vel, _) = Particle::get_attributes(&w.particles().buf);
                shader.draw(&mut context, DrawMode::Points, n, den, mat, ref_pos, pos, vel);

            }
        );

        let trans = !lighting;

        unsafe {
            gl::Viewport(80*2,0,win_h as i32,win_h as i32);
            gl::Disable(gl::CULL_FACE);

            gl::Enable(0x8861);
            gl::Enable(gl::BLEND);

            if trans {
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
