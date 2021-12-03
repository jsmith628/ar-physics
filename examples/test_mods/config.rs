
use super::*;

use toml::value::{Table, Array};

#[derive(Clone, Default)]
pub struct Light {
    pub pos: vec3,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub color: vec3,
}

#[derive(Clone, Default)]
pub struct SimulationObject<'a> {
    pub name: &'a str,
    pub colors: [vec4; 3],
    pub region: MaterialRegion,
}

#[derive(Clone, Default)]
pub struct InteractiveObject {
    pub colors: [vec4; 3],
    pub relative: bool,
    pub region: MaterialRegion,
}

#[derive(Clone, Default)]
pub struct Config<'a> {

    pub title: &'a str,
    pub lighting: bool,

    pub subticks: u32,
    pub h: f32,
    pub render_h: f32,
    pub dt: f32,
    pub alpha: f32,
    pub g: f32,

    pub min: vec4,
    pub dim: vec4,

    pub render_factor: f32,
    pub light: Light,

    pub objects: Vec<SimulationObject<'a>>,
    pub on_click: Vec<InteractiveObject>,
    pub interactions: Vec<Vec<Option<MatInteraction>>>,

    pub integrator: &'a str,

    pub camera_pos: Vec<f64>,
    pub scale: f64,
    pub rot: f64

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

fn get_packing(region: &Value, h: f32) -> f32 {
    as_float_or(region, "packing", as_float_or(region, "spacing", 0.5*h as f64)/h as f64) as f32
}

fn get_colors(mat: &Value) -> [vec4;3] {
    let color = as_vec4_or(&mat, "color", [0.0,0.0,0.0,1.0].into());
    [
        color,
        as_vec4_or(&mat, "color_low_density", color),
        as_vec4_or(&mat, "color_high_density", color)
    ]
}

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

impl<'a> Config<'a> {

    pub fn parse(config: &'a Value) -> Config<'a> {

        let mut cfg = Config::default();

        cfg.title = as_str_or(&config, "name", "Fluid Test");
        cfg.lighting = as_bool_or(&config, "lighting", config.as_table().unwrap().get("light_pos").is_some());

        cfg.subticks = as_int_or(&config, "subticks", 1) as u32;
        cfg.h = as_float_or(&config, "kernel_radius", 1.0/64.0) as f32;
        cfg.dt = as_float_or(&config, "time_step", 0.01) as f32;
        cfg.alpha = as_float_or(&config, "artificial_viscocity", 50.0) as f32;
        cfg.g = as_float_or(&config, "gravity", 1.0) as f32;

        cfg.min = as_vec4_or(&config, "min", [-1.0,-1.0,0.0,0.0].into());
        cfg.dim = as_vec4_or(&config, "dim", [2.0,2.0,0.0,0.0].into());


        cfg.render_h = cfg.h * as_float_or(&config, "particle_render_factor", 1.0) as f32;

        cfg.light = Light {
            pos: as_vec3_or(&config, "light_pos", [-3.0,3.0,-3.0].into()),
            ambient: as_float_or(&config, "ambient_brightness", 1.0) as f32,
            diffuse: as_float_or(&config, "diffuse_brightness", 10.0) as f32,
            specular: as_float_or(&config, "specular_brightness", 0.0) as f32,
            color: as_vec3_or(&config, "light_color", [1.0,1.0,1.0].into())
        };

        let h = cfg.h;
        let alpha = cfg.alpha;

        let mut interaction_map = HashMap::new();

        if let Some(Value::Table(boundaries)) = config.get("boundaries") {
            for (name, immobile) in boundaries {
                let packing = get_packing(&immobile, h);
                let friction = as_float_or(&immobile, "friction", 0.0) as f32;

                let region = parse_region_from_mat(immobile.as_table().unwrap());
                let colors = get_colors(&immobile);
                let mat = MaterialRegion::new(region, packing, Material::new_immobile(friction));

                cfg.objects.push(
                    SimulationObject {
                        name: name.as_str(),
                        colors: colors,
                        region: mat
                    }
                );

                get_interactions(name.as_str(), &immobile, h as f64, &mut interaction_map);
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

                //the angular velocity can be 6 elements since 4D ang velocity takes 6 elements
                let mut ang_vel = [0.0;6];
                if let Some(arr) = table.get("angular_velocity") {
                    for (i, val) in arr.as_array().unwrap().into_iter().enumerate().take(6) {
                        ang_vel[i] = val.as_float().unwrap() as f32;
                    }
                }

                let colors = get_colors(&obj);

                let mat_region = MaterialRegion::with_vel(
                    region,
                    get_packing(&obj, h),
                    mat,
                    move |p| {
                        let r = [p[0]-center[0], p[1]-center[1], p[2]-center[2], p[3]-center[3]];
                        let w = ang_vel;

                        //this general case from good ol' geometric algebra
                        [
                            w[1]*r[2] - w[2]*r[1] - w[3]*r[3] + vel[0],
                            w[2]*r[0] - w[0]*r[2] + w[4]*r[3] + vel[1],
                            w[0]*r[1] - w[1]*r[0] - w[5]*r[3] + vel[2],
                            w[3]*r[0] - w[4]*r[1] + w[5]*r[3] + vel[3],
                        ].into()
                    }
                );

                if !table.get("on_click").is_some() {

                    cfg.objects.push(
                        SimulationObject {
                            name: name.as_str(),
                            colors: colors,
                            region: mat_region
                        }
                    );

                    get_interactions(name.as_str(), &obj, h as f64, &mut interaction_map);
                } else {
                    let relative = table.get("on_click").unwrap().as_str().unwrap() == "Relative";
                    cfg.on_click.push(
                        InteractiveObject {
                            colors: colors,
                            relative: relative,
                            region: mat_region
                        }
                    )
                }

            }
        }

        cfg.integrator = as_str_or(&config, "integrator", "verlet");

        for i in 0..cfg.objects.len() {
            let name1 = cfg.objects[i].name;
            let mut sublist = Vec::new();
            for j in 0..cfg.objects.len() {
                let name2 = cfg.objects[j].name;
                sublist.push(
                    (match interaction_map.get(&(name1,name2)) {
                        Some(a) => Some(a),
                        None => interaction_map.get(&(name2,name1))
                    }).map(|a| *a)
                );
            }
            cfg.interactions.push(sublist);
        }

        cfg.camera_pos = as_float_vec_or(&config, "view_pos", vec![0.0,0.0]);
        cfg.scale = as_float_or(&config, "view_scale", 1.0);
        cfg.rot = as_float_or(&config, "view_angle", 0.0).to_radians();

        cfg

    }

}
