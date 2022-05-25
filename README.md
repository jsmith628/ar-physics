
An SPH-based Fluid/Solid simulation based in Rust and GLSL and designed primarily
for real-time operation on the GPU

This is an old project of mine developed back from around 2018 to 2020 as
part of an undergraduate independent study with the Patankar Fluid Simulation
Lab. Though it was created under the supervision and wonderful guidance of the
lab, every line of code in it is my own.

At this point, I no longer intend to make any more updates, so the code is being
hosted primarily for archival purposes (and to admittedly show off some code I am
quite proud of).

Whether you are an employer needing to see my programming style or you're
just looking for an implementation of simulations of continuum
mechanics to reference, feel free to poke around in the code to glean whatever
you can from it. Just don't expect a university project to be full professional
simulation software by any means.

# How to Run

To compile and run the simulation, make sure [`cargo`](1) and Rust and is
installed on your system, and simply run the `test.rs` example while providing a
`.toml` to configure the simulation
```bash
cargo run --release --example test -- "examples/your-example.toml"
```

Provided in the `examples/` directory is a large set of premade examples and
tests that should work out of the box. (save for a couple, like
`examples/wave_3d.toml`, that were designed to be recorded instead of played
in real-time).

Additionally, `examples/test.sh` provides a simple script for selecting between
the noteworthy examples using a `POSIX` shell.

Finally, all of the runtime flags for the test can be queried with:
```bash
cargo run --release --example test -- --help
```
These options include capping the framerate and tickrate, changing the
window size, etc.

# How to Record

The `test` example also allows for direct recording of each frame of the
simulation. If the `-r` or `--record` option is used, the program will
output each frame as a raw `RGB24` image directly to `stdout`. This output
can then be directed into another program (eg. `ffmpeg`) to convert the
data into a more suitable compressed format:
```bash
cargo run --release --example test -- \
  /examples/wave_3d.toml --record --fps 60 --record-width 1920 --record-height 1080 | \
  ffmpeg -f rawvideo -r 60 -s 1920x1080 -i - -vf vflip examples/recordings/out.mp4
```


[1]: https://doc.rust-lang.org/cargo/getting-started/installation.html
