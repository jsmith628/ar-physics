#![feature(maybe_uninit_ref)]
#![recursion_limit="2048"]

#[macro_use]
extern crate gl_struct;
extern crate glfw;
extern crate toml;
extern crate stl_io;
extern crate rayon;

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

                // frag_color.a = 1;

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

pub struct Mesh {
    mesh:IndexedMesh
}

pub struct Surface {
    mesh:IndexedMesh,
    border: f32
}

fn point_bound(verts: &Vec<Vertex>) -> AABB {
    let mut min = [verts[0][0], verts[0][1], verts[0][2], 0.0];
    let mut max = min;

    for v in verts.iter() {
        for i in 0..3 {
            min[i] = v[i].min(min[i]);
            max[i] = v[i].max(max[i]);
        }
    }

    AABB::from_min_max(min.into(),max.into())
}

impl Region for Mesh {
    fn bound(&self) -> AABB { point_bound(&self.mesh.vertices) }

    fn contains(&self, p: vec4) -> bool {

        //if we have a 4D point, we're not in the mesh
        if p.value[3] != 0.0 { return false; }

        //because we don't actually have a good dot product operation
        // fn dot(v1:&[f32], v2: &[f32]) -> f32 { v1[0]*v2[0] + v1[1]*v2[1] }

        self.mesh.faces.iter().map(
            |f| [
                self.mesh.vertices[f.vertices[0]],
                self.mesh.vertices[f.vertices[1]],
                self.mesh.vertices[f.vertices[2]]
            ]
        ).map(
            |verts| {
                let above = {
                    verts[0][2] >= p.value[2] ||
                    verts[1][2] >= p.value[2] ||
                    verts[2][2] >= p.value[2]
                };

                if above {
                    let (a,b,c) = (
                        [verts[0][0],verts[0][1]],
                        [verts[1][0],verts[1][1]],
                        [verts[2][0],verts[2][1]]
                    );

                    let (s1,s2,d) = (
                        [b[0]-a[0],b[1]-a[1]],
                        [c[0]-a[0],c[1]-a[1]],
                        [p.value[0]-a[0],p.value[1]-a[1]]
                    );

                    let mut a = s1[0]*s2[1] - s1[1]*s2[0];
                    let s = a.signum() * (d[0]*s2[1] - d[1]*s2[0]);
                    let t = a.signum() * (s1[0]*d[1] - s1[1]*d[0]);
                    a = a.abs();

                    s>=0.0 && s<=a && t>=0.0 && t<=a && s+t<=a

                } else {
                    false
                }
            }
        ).fold(false, |b1,b2| b1^b2)
    }
}

impl Region for Surface {

    fn bound(&self) -> AABB {
        let mut bound = point_bound(&self.mesh.vertices);
        for i in 0..3 {
            bound.min[i] -= self.border;
            bound.dim[i] += self.border*2.0;
        }
        bound
    }

    fn contains(&self, p: vec4) -> bool {

        fn dot(v1: &[f32], v2: &[f32]) -> f32 {
            v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        }

        fn cross(v1: &[f32], v2: &[f32]) -> [f32;3] {
            [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]
        }

        fn sub(v1: &[f32], v2: &[f32]) -> [f32;3] { [v1[0]-v2[0],v1[1]-v2[1],v1[2]-v2[2]] }

        for v in self.mesh.vertices.iter() {
            let d = sub(&p.value, v);
            if dot(&d, &d) <= self.border*self.border { return true; }
        }


        for f in self.mesh.faces.iter() {
            let n = f.normal;

            let (a,b,c) = (
                self.mesh.vertices[f.vertices[0]],
                self.mesh.vertices[f.vertices[1]],
                self.mesh.vertices[f.vertices[2]]
            );

            let d = sub(&p.value, &a);

            if dot(&d, &n).abs() <= dot(&n, &n).sqrt()*self.border {

                let (s1, s2, s3, s4) = (sub(&b,&a), sub(&c,&a), sub(&c,&b), sub(&a,&b));

                let d2 = sub(&p.value, &b);

                let (n1, n2, n3) = (cross(&s1, &n), cross(&s2, &n), cross(&s3, &n));

                if
                    dot(&d, &n1).signum() == dot(&s2, &n1).signum() &&
                    dot(&d, &n2).signum() == dot(&s1, &n2).signum() &&
                    dot(&d2, &n3).signum() == dot(&s4, &n3).signum()
                { return true; }
            }
        }

        return false;

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

        let config = {

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
        if record {window.set_resizable(false);}

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

        let mut mat_number = 0;

        let mut list = Vec::new();
        let mut names = Vec::new();
        let mut on_click = Vec::new();
        let mut interaction_map = HashMap::new();

        {

            use toml::value::{Table, Array};

            fn parse_region_from_mat(region: &Table) -> Arc<dyn Region+Send+Sync> {

                type Reg = Arc<dyn Region+Send+Sync>;

                fn fold_array<F:Fn(Reg,Reg) -> Reg>(arr: &Array, f:F) -> Reg {
                    let mut iter = arr.into_iter().map(|s| parse_region(s.as_table().unwrap()));
                    let first = iter.next().unwrap();
                    iter.fold(first, f)
                }

                fn parse_region(table: &Table) -> Reg {

                    let base = parse_base(table);
                    let reg = Value::Table(table.clone());

                    if table.get("translation").is_some() || table.get("scale").is_some() || table.get("rotation").is_some() {

                        let trans = as_vec4_or(&reg, "translation", [0.0,0.0,0.0,0.0].into());

                        let rot = match table.get("rotation") {
                            Some(Value::Float(f)) => [0.0,0.0,1.0,*f as f32],
                            _ => as_vec4_or(&reg, "rotation", [0.0,0.0,1.0,0.0].into()).value
                        };

                        let scale = match table.get("scale") {
                            Some(Value::Float(f)) => [*f as f32;4],
                            Some(Value::Array(arr)) => {
                                let mut val = [1.0;4];
                                for i in 0..(arr.len().min(4)) {val[i] = arr[i].as_float().unwrap() as f32;}
                                val
                            },
                            _ => [1.0;4]
                        };

                        let c = rot[3].to_radians().cos();
                        let s = rot[3].to_radians().sin();
                        let u_l = (rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2]).sqrt();
                        let u = [rot[0]/u_l, rot[1]/u_l, rot[2]/u_l];

                        let mut mat = [
                            [c+u[0]*u[0]*(1.0-c), u[0]*u[1]*(1.0-c)+u[2]*s, u[0]*u[2]*(1.0-c)-u[1]*s, 0.0],
                            [u[1]*u[2]*(1.0-c)-u[1]*s, c+u[1]*u[1]*(1.0-c), u[1]*u[2]*(1.0-c)+u[0]*s, 0.0],
                            [u[0]*u[2]*(1.0-c)+u[1]*s, u[1]*u[2]*(1.0-c)-u[0]*s, c+u[2]*u[2]*(1.0-c), 0.0],
                            [0.0,0.0,0.0,1.0]
                        ];


                        for i in 0..4 {
                            for j in 0..4 {
                                mat[i][j] *= scale[i];
                            }
                        }

                        Arc::new(Transformed(base,mat.into(),trans))

                    } else {
                        base
                    }
                }

                fn parse_base(region: &Table) -> Reg {
                    if let Some(Value::String(path)) = region.get("model") {
                        if let Some(border) = region.get("border").and_then(|b| b.as_float()) {
                            Arc::new(
                                Surface{
                                    mesh: read_stl(&mut File::open(path).unwrap()).unwrap(),
                                    border: border as f32
                                }
                            )
                        } else {
                            Arc::new(Mesh{mesh:read_stl(&mut File::open(path).unwrap()).unwrap()})
                        }
                    } else {
                        let boxed;
                        let mut dim;
                        let mut min = [0.0,0.0,0.0,0.0].into();
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
                                Arc::new(AABB {min: min, dim: dim}.border(depth))
                            } else {
                                Arc::new(AABB {min: min, dim: dim})
                            }
                        } else {
                            for i in 0..4 {
                                dim[i] *= 0.5;
                                min[i] += dim[i];
                            }
                            if let Some(depth) = border {
                                Arc::new(AABE {center: min, radii: dim}.border(depth))
                            } else {
                                Arc::new(AABE {center: min, radii: dim})
                            }
                        }
                    }

                }


                if let Some(Value::Array(arr)) = region.get("union") {
                    fold_array(arr, |s1,s2| Arc::new(Union(s1,s2)))
                } else if let Some(Value::Array(arr)) = region.get("difference") {
                    fold_array(arr, |s1,s2| Arc::new(Difference(s1,s2)))
                } else if let Some(Value::Array(arr)) = region.get("intersection") {
                    fold_array(arr, |s1,s2| Arc::new(Intersection(s1,s2)))
                } else if let Some(Value::Table(table)) = region.get("region") {
                    parse_region(table)
                } else {
                    panic!("Material has no table named region or doesn't have union, difference, or intersection arrays");
                }
            }

            fn parse_interation(table: &Value, h: f64) -> MatInteraction {
                MatInteraction {
                    strength: as_float_or(table, "contact_strength", 10000.0) as f32,
                    radius: as_float_or(table, "contact_radius", as_float_or(table, "contact_factor", 2.0)*h) as f32,
                    friction: as_float_or(table, "friction", 0.0) as f32,
                    dampening: as_float_or(table, "dampening", 0.0) as f32,

                    potential: {
                        match table.as_table().unwrap().get("potential").map(|a| a.as_str().unwrap()) {
                            Some("Zero") | Some("zero") => ContactPotental::Zero,
                            Some("Constant") | Some("constant") => ContactPotental::Constant,
                            Some("Linear") | Some("linear") => ContactPotental::Linear,
                            Some("Tait") | Some("tait") => ContactPotental::Tait,
                            Some("Hertz") | Some("hertz") => ContactPotental::Hertz,
                            Some("LennardJones") | Some("Lennard-Jones") | Some("Lennard Jones") |
                                Some("lennard jones") | Some("lennard-jones") => ContactPotental::LennardJones,

                            _ => ContactPotental::Linear,
                        }
                    } as u32
                }

            }

            fn get_interactions<'a>(
                name: &'a str, obj: &'a Value, h: f64, map: &mut HashMap<(&'a str,&'a str), MatInteraction>
            ) {
                for (name2, interaction) in obj.as_table().unwrap() {
                    if interaction.as_table().is_some() {
                        let mut proper_name = name2.as_str() != "region";
                        proper_name &= name2.as_str() != "union";
                        proper_name &= name2.as_str() != "intersection";
                        proper_name &= name2.as_str() != "difference";

                        if proper_name {
                            map.insert((name, name2.as_str()) , parse_interation(interaction, h as f64));
                        }
                    }
                }
            }

            fn set_colors_and_den(mat:&Value, shader: &mut ParticleShader::Program, id:usize, den: f32) {
                let color = as_vec4_or(&mat, "color", [0.0,0.0,0.0,1.0].into());
                shader.c2[id] = color;
                shader.c1[id] = as_vec4_or(&mat, "color_low_density", color);
                shader.c3[id] = as_vec4_or(&mat, "color_high_density", color);

                shader.densities[id] = den;
            }

            fn get_packing(region: &Value, h: f32) -> f32 {
                as_float_or(region, "packing", as_float_or(region, "spacing", 0.5*h as f64)/h as f64) as f32
            }

            if let Some(Value::Table(boundaries)) = config.get("boundaries") {
                for (name, immobile) in boundaries {
                    let packing = get_packing(&immobile, h);
                    let friction = as_float_or(&immobile, "friction", 0.0) as f32;

                    let region = parse_region_from_mat(immobile.as_table().unwrap());

                    list.push(MaterialRegion::new(region, packing, Material::new_immobile(friction)));
                    names.push(name.as_str());
                    get_interactions(name.as_str(), &immobile, h as f64, &mut interaction_map);

                    set_colors_and_den(&immobile, &mut shader, mat_number, 1.0);
                    mat_number += 1;
                }
            }

            if let Some(Value::Table(objects)) = config.get("objects") {
                for (name, obj) in objects {
                    let table = obj.as_table().unwrap();
                    let mut mat = Material::default();
                    mat.start_den = as_float_or(&obj, "start_density", as_float_or(&obj, "density", 1.0)) as f32;
                    mat.target_den = as_float_or(&obj, "target_density", as_float_or(&obj, "density", 1.0)) as f32;
                    mat.sound_speed = as_float_or(&obj, "speed_of_sound", 0.0) as f32;
                    mat.visc = as_float_or(&obj, "viscocity", as_float_or(&obj, "friction", 0.0)) as f32;
                    mat.bulk_visc = as_float_or(&obj, "bulk_viscocity", (alpha*h*mat.sound_speed) as f64) as f32;
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
                        Some("Constant") | Some("constant") => StateEquation::Constant,
                        Some(s) => panic!("Invalid state equation: {}", s)
                    } as u32;

                    mat.strain_order = as_int_or(&obj, "strain_order", 2) as i32;
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

                    let mat_region = MaterialRegion::with_vel(
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
                    );

                    if !table.get("on_click").is_some() {
                        list.push(mat_region);
                        names.push(name.as_str());
                        get_interactions(name.as_str(), &obj, h as f64, &mut interaction_map);
                        set_colors_and_den(&obj, &mut shader, mat_number, mat.target_den);
                        mat_number += 1;
                    } else {
                        let relative = table.get("on_click").unwrap().as_str().unwrap() == "Relative";
                        on_click.push(
                            (mat_region, relative, as_vec4_or(&obj, "color", [0.0,0.0,0.0,1.0].into()))
                        );
                    }

                }
            }

        };

        let integrator: Box<dyn VelIntegrates<_, _>> = {
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

        let mut interaction_lists = Vec::new();

        for i in 0..list.len() {
            let name1 = names[i];
            let mut sublist = Vec::new();
            for j in 0..list.len() {
                let name2 = names[j];
                sublist.push(
                    (match interaction_map.get(&(name1,name2)) {
                        Some(a) => Some(a),
                        None => interaction_map.get(&(name2,name1))
                    }).map(|a| *a)
                );
            }
            interaction_lists.push(sublist);
        }

        let interactions = interaction_lists.iter().map(|l| &l[0..]).collect::<Vec<_>>();

        if record {unsafe {ar_physics::LOGGING = false}; }

        let mut engine = Engine::new();
        engine.add_component(
            "world",
            if tps < 0.0 {ConstantTimer::new_uncapped()} else {ConstantTimer::from_tps(tps)},
            FluidSim::new(&gl_provider,
                &list[0..], &interactions[0..], AABB{min:min, dim:dim}, h, integrator, dt, subticks, g, alpha
            ).unwrap()
        );

        let world = engine.get_component::<FluidSim>("world").unwrap();

        let window1 = Arc::new(RefCell::new(window));
        let window2 = window1.clone();

        let camera_pos = as_float_vec_or(&config, "view_pos", vec![0.0,0.0]);
        let scale = as_float_or(&config, "view_scale", 1.0);
        let mut rot = as_float_or(&config, "view_angle", 0.0).to_radians();

        let (mut x, mut y) = window1.borrow().get_cursor_pos();
        let (mut l_pressed, mut m_pressed, mut r_pressed) = (false, false, false);
        let (mut trans_x, mut trans_y) = (-camera_pos[0],-camera_pos[1]);

        let mut pixels = if record {
            Some(vec![0u8; 3usize * rec_w as usize * rec_h as usize].into_boxed_slice())
        } else {None};

        let [color,depth] = unsafe {
            let mut rb = MaybeUninit::<[GLuint;2]>::uninit();
            gl::GenRenderbuffers(2, &mut rb.get_mut()[0] as *mut GLuint);
            rb.assume_init()
        };

        let fb = unsafe {
            let mut fb = MaybeUninit::uninit();
            gl::GenFramebuffers(1, fb.get_mut());
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
                    for (region, relative, color) in on_click.iter() {
                        shader.densities[mat_number] = region.mat.target_den;
                        shader.c1[mat_number] = *color;
                        shader.c2[mat_number] = *color;
                        shader.c3[mat_number] = *color;

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

        let trans = !lighting;

        unsafe {
            gl::Viewport(80*2,0,win_h as i32,win_h as i32);
            gl::Disable(gl::CULL_FACE);
            gl::Disable(gl::RASTERIZER_DISCARD);

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
