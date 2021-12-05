#![feature(maybe_uninit_ref)]
#![recursion_limit="2048"]

#[macro_use]
extern crate gl_struct;
extern crate glfw;
extern crate toml;
extern crate stl_io;

extern crate ar_physics;
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
use std::time::*;
use std::io::Read;
use std::fs::File;
use std::collections::HashMap;
use std::thread::sleep;

use ar_physics::soft_body::*;
use ar_physics::soft_body::material_region::*;
use ar_physics::soft_body::particle_state::*;

use numerical_integration::*;

pub use self::test_mods::*;
pub use self::test_mods::Surface;

mod test_mods;

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

        let mut renderer =  Renderer::init(&cfg, win_w, win_h, rec_w, rec_h, record);
        let window = renderer.window();

        let mut sim = {
            FluidSim::new(
                renderer.gl(),
                &objects[0..], &interactions[0..],
                AABB{min:cfg.min, dim:cfg.dim}, cfg.h,
                integrator, cfg.dt, cfg.subticks,
                cfg.g, cfg.alpha
            ).unwrap()
        };
        sim.init();

        sleep(Duration::from_millis(1000));

        //the main simulation loop
        let (dt, df) = (1.0 / tps, 1.0 / fps);
        let (mut last_tick, mut last_frame) = (Instant::now(), Instant::now());
        while !window.borrow().should_close() {

            //note that if dt or df is negative, then the checks will always come out true

            let now = Instant::now();
            if (now - last_tick).as_secs_f64() >= dt {
                last_tick = now;
                sim.update();
            }

            let now = Instant::now();
            if (now - last_frame).as_secs_f64() >= df {
                last_frame = now;
                renderer.render(&mut sim);
            }

            //sleep for a little bit so we don't just waste cpu in a loop
            let now = Instant::now();
            let (d1, d2) = (now - last_tick, now - last_frame);

            //gotta make sure there's no overflow
            if dt > d1.as_secs_f64() && df > d2.as_secs_f64() {
                sleep((Duration::from_secs_f64(dt)-d1).min(Duration::from_secs_f64(df)-d2) / 2);
            }

        }

    } else {
        println!("No config provided. Exiting.");
    }



}
