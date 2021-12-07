#![recursion_limit="2048"]

extern crate glfw;
extern crate toml;
extern crate stl_io;
extern crate clap;

#[macro_use]
extern crate gl_struct;
extern crate ar_physics;
extern crate numerical_integration;

use toml::Value;
use clap::*;
use stl_io::*;

use std::io::*;
use std::mem::*;
use std::time::*;

use std::io::Read;
use std::fs::File;
use std::collections::HashMap;
use std::thread::sleep;
use std::sync::Arc;
use std::cell::RefCell;

use gl_struct::*;
use gl::types::*;
use gl_struct::glsl_type::{vec4, vec3};

use ar_physics::soft_body::*;
use ar_physics::soft_body::material_region::*;
use ar_physics::soft_body::particle_state::*;

use numerical_integration::*;

pub use self::test_mods::*;
pub use self::test_mods::Surface;

mod test_mods;

fn main() {

    let app = {
        app_from_crate!()
        .setting(AppSettings::ColorAuto)
        .setting(AppSettings::GlobalVersion)
        .setting(AppSettings::DisableHelpSubcommand)
        .arg(
            Arg::with_name("tps")
            .short("t")
            .long("tps")
            .takes_value(true)
            .value_name("RATE")
            .default_value("-1")
            .set(ArgSettings::AllowLeadingHyphen)
            .help("The target number of ticks per second")
            .long_help(
                "The target number of ticks per second.\n\
                 A negative value will result in an unlocked tick rate."
            )
        ).arg(
            Arg::with_name("fps")
            .short("f")
            .long("fps")
            .takes_value(true)
            .value_name("RATE")
            .default_value("-1")
            .set(ArgSettings::AllowLeadingHyphen)
            .help("The target number of frames per second")
            .long_help(
                "The target number of frames per second.\n\
                 A negative value will result in an unlocked frame rate."
            )
        ).arg(
            Arg::with_name("window width")
            .short("w")
            .long("width")
            .takes_value(true)
            .value_name("WIDTH")
            .default_value("960")
            .help("The initial width of the application window")
        ).arg(
            Arg::with_name("window height")
            .short("h")
            .long("height")
            .takes_value(true)
            .value_name("HEIGHT")
            .default_value("720")
            .help("The initial height of the application window")
        ).arg(
            Arg::with_name("record")
            .short("r")
            .long("record")
            .help("Causes the application to make an additional render offscreen and print each frame to stdout")
        ).arg(
            Arg::with_name("recording width")
            // .short("rw")
            .long("record-width")
            .takes_value(true)
            .value_name("WIDTH")
            .help("The width in pixels of the application recording")
        ).arg(
            Arg::with_name("recording height")
            // .short("rh")
            .long("record-height")
            .takes_value(true)
            .value_name("HEIGHT")
            .help("The height in pixels of the application recording")
        ).arg(
            Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Causes debug information to be printed")
        ).arg(
            Arg::with_name("example config")
            .takes_value(true)
            .value_name("CONFIG")
            .help("A toml file specifying the settings for the simulation")
        )
    };

    //
    //Parse cli args
    //

    let matches = app.get_matches();

    let config_loc = matches.value_of("example config");

    let tps = matches.value_of("tps").unwrap().parse::<f64>().expect("--tps requires a floating point argument");
    let fps = matches.value_of("fps").unwrap().parse::<f64>().expect("--fps requires a floating point argument");

    let win_w = matches.value_of("window width").unwrap().parse::<u32>().expect("--width requires a positive integer argument");
    let win_h = matches.value_of("window height").unwrap().parse::<u32>().expect("--height requires a positive integer argument");
    let rec_w = matches.value_of("recording width").map(
        |s| s.parse::<u32>().expect("--record-width requires a positive integer argument")
    ).unwrap_or(win_w);
    let rec_h = matches.value_of("recording height").map(
        |s| s.parse::<u32>().expect("--record-height requires a positive integer argument")
    ).unwrap_or(win_h);

    let record = matches.is_present("record");
    let verbose = matches.is_present("verbose");

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

        //reformat the object and interactions list. This could probably be a little more efficient
        let interactions = cfg.interactions.iter().map(|l| &l[0..]).collect::<Vec<_>>();
        let objects = cfg.objects.clone().into_iter().map(|x| x.region).collect::<Vec<_>>();

        //enable debug logging if we aren't recording and -v was used
        unsafe {ar_physics::LOGGING = !record && verbose};

        //initialize the window and renderer
        let mut renderer =  Renderer::init(&cfg, win_w, win_h, rec_w, rec_h, record);
        let window = renderer.window();

        //pass in all the simulation configuration and init the simulation
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

        //ngl... I don't remember why this is here, but I don't wanna touch it for now...
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

            //
            //Sleep for a little bit so we don't just waste cpu in a loop
            //

            let now = Instant::now();
            let (d1, d2) = (now - last_tick, now - last_frame);

            //gotta make sure there's no overflow
            if dt > d1.as_secs_f64() && df > d2.as_secs_f64() {
                //sleep for half the time until the next tick or frame
                //the `/2` part is so that we don't accidentally take too much time sleeping
                //it's a little crude, but it works
                sleep((Duration::from_secs_f64(dt)-d1).min(Duration::from_secs_f64(df)-d2) / 2);
            }

        }

    } else {
        println!("No config provided. Exiting.");
    }



}
