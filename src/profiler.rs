
use std::collections::HashMap;
use std::time::*;

use gl_struct::gl;


const PROFILE_BARRIER: bool = false;

pub(crate) struct Profiler {
    start_of_frame: Instant,
    start_of_current: Instant,
    current: Option<String>,
    segments: HashMap<String, Duration>
}

impl Profiler {

    pub fn new() -> Self {
        Profiler {
            start_of_frame: Instant::now(),
            start_of_current: Instant::now(),
            current: None,
            segments: HashMap::new()
        }
    }

    fn add_duration(segments: &mut HashMap<String, Duration>, name:&String, dt: Duration) {
        if let Some(d) = segments.get_mut(name) {
            *d += dt;
            return;
        }
        segments.insert(name.clone(), dt);
    }

    pub fn new_frame(&mut self) -> (Duration, HashMap<String, Duration>) {
        // println!("New frame!");
        if PROFILE_BARRIER {unsafe { gl::Flush(); gl::Finish(); } }
        //get the duration of the frame
        let time = Instant::now();
        let frame_length = time - self.start_of_frame;

        //end the current segment as well
        if let Some(name) = &self.current {
            Self::add_duration(&mut self.segments, name, time - self.start_of_current);
            self.current = None;
        }

        //swap out the segment data with an empty map
        let mut temp = HashMap::with_capacity(self.segments.capacity());
        ::std::mem::swap(&mut self.segments, &mut temp);

        self.start_of_frame = Instant::now();

        (frame_length, temp)
    }

    pub fn end_segment(&mut self) -> Option<Duration> {
        if PROFILE_BARRIER {unsafe { gl::Flush(); gl::Finish(); } }
        let result = match &self.current {
            Some(name) => {
                // println!("Ending {}: ", name);
                let seg_length = Instant::now() - self.start_of_current;
                Self::add_duration(&mut self.segments, name, seg_length);
                Some(seg_length)
            },
            None => None
        };
        self.current = None;
        result
    }

    pub fn new_segment(&mut self, name: String) -> Option<Duration> {
        let seg_length = self.end_segment();
        // println!("Starting {}: ", name);
        self.current = Some(name);
        self.start_of_current = Instant::now();
        seg_length
    }

}
