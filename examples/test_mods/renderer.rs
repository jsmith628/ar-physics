
use super::*;
use glfw::*;

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

pub struct Renderer {

    glfw: Glfw,
    window: Arc<RefCell<Window>>,
    gl: gl_struct::GLProvider,
    context: gl_struct::Context,
    shader: ParticleShader::Program,

    pixels: Option<Box<[u8]>>,
    fb: GLuint,
    rec_w: GLsizei,
    rec_h: GLsizei,

    x: f64,
    y: f64,
    rot: f64,
    scale: f64,
    trans_x: f64,
    trans_y: f64,

    l_pressed: bool,
    r_pressed: bool,
    m_pressed: bool,

    on_click: Vec<InteractiveObject>

}

impl Renderer {

    pub fn window(&self) -> Arc<RefCell<Window>> { self.window.clone() }
    pub fn gl(&self) -> &GLProvider { &self.gl }

    pub fn init(cfg: &Config, win_w: u32, win_h: u32, rec_w: u32, rec_h: u32, record: bool) -> Renderer {

        // println!("{} {} {} {}", win_w, win_h, rec_w, rec_h);

        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        let mut window = glfw.create_window(win_w, win_h, cfg.title, glfw::WindowMode::Windowed).unwrap().0;
        if record {window.set_resizable(false);}

        glfw::Context::make_current(&mut window);
        glfw.set_swap_interval(glfw::SwapInterval::None);

        let window = Arc::new(RefCell::new(window));

        let gl = unsafe {
            GLProvider::load(|s| ::std::mem::transmute(glfw.get_proc_address_raw(s)))
        };
        let mut context = gl_struct::Context::init(&gl);
        let mut shader = ParticleShader::init(&gl).unwrap();

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

        let size = 3usize * rec_w as usize * rec_h as usize;
        let mut pixels = if record {
            Some(vec![0u8; size].into_boxed_slice())
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

        //
        //Some global GL settings
        //

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

        let (x, y) = window.borrow().get_cursor_pos();

        Renderer {
            glfw, window, gl, context, shader,

            pixels, fb,
            rec_w: rec_w as GLsizei,
            rec_h: rec_h as GLsizei,

            x, y,
            rot: cfg.rot,
            scale: cfg.scale,
            trans_x: -cfg.camera_pos[0],
            trans_y: -cfg.camera_pos[1],

            l_pressed: false,
            r_pressed: false,
            m_pressed: false,

            on_click: cfg.on_click.clone()
        }

    }


    pub fn render(&mut self, w: &mut FluidSim) {

        //
        //grab the last frame and write it to stdout
        //

        if let Some(pixels) = self.pixels.as_mut() {
            use std::io::Write;

            unsafe {

                gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.fb);
                gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

                gl::Flush();
                gl::Finish();

                gl::ReadnPixels(
                    0,0,
                    self.rec_w as GLsizei, self.rec_h as GLsizei,
                    gl::RGB,gl::UNSIGNED_BYTE,
                    pixels.len() as GLsizei,
                    &mut pixels[0] as *mut u8 as *mut gl::types::GLvoid
                );

            }

            std::io::stdout().write_all(&**pixels).unwrap();

        }

        //
        //Use the motion to change the translation
        //

        //note, each thing is a column, not row
        {
            // let s = cfg.scale as f32;
            let t = self.rot as f32;
            *self.shader.trans = [
                [t.cos(),              0.0,                  t.sin(),                     0.0],
                [ 0.0,                 1.0,                  0.0,                         0.0],
                [-t.sin(),             0.0,                  t.cos(),                     0.0],
                [self.trans_x as f32, -self.trans_y as f32, -self.scale as f32+1.0,       1.0],
            ].into();
        }

        //
        //Input
        //

        let (new_x, new_y) = self.window.borrow().get_cursor_pos();
        self.l_pressed = match self.window.borrow().get_mouse_button(glfw::MouseButton::Button1) {
            glfw::Action::Press => true,
            glfw::Action::Release => false,
            _ => self.l_pressed
        };

        self.r_pressed = match self.window.borrow().get_mouse_button(glfw::MouseButton::Button2) {
            glfw::Action::Press => true,
            glfw::Action::Release => false,
            _ => self.r_pressed
        };

        let m = self.m_pressed;
        self.m_pressed = match self.window.borrow().get_mouse_button(glfw::MouseButton::Button3) {
            glfw::Action::Press => true,
            glfw::Action::Release => false,
            _ => self.m_pressed
        };

        if self.l_pressed {self.rot += (new_x - self.x)*0.01;}
        // if m_pressed { scale += (new_x - x)*0.01;}
        if self.r_pressed {
            self.trans_x += (new_x - self.x)*0.005;
            self.trans_y += (new_y - self.y)*0.005;
        }
        self.x = new_x;
        self.y = new_y;

        let (size_x, size_y) = self.window.borrow().get_size();
        let min_size = size_x.min(size_y);

        if !m && self.m_pressed {
            for obj in self.on_click.iter() {

                let (region, relative, colors) = (&obj.region, obj.relative, obj.colors);

                let mat_number = w.num_materials();

                self.shader.densities[mat_number] = region.mat.target_den;
                self.shader.c1[mat_number] = colors[0];
                self.shader.c2[mat_number] = colors[1];
                self.shader.c3[mat_number] = colors[2];

                let offset = match relative {
                    true => Some([
                        2.0*((self.x as f32 - size_x as f32 /2.0) / min_size as f32),
                        2.0*((size_y as f32/2.0 - self.y as f32) / min_size as f32),0.0,0.0
                    ].into()),
                    false => None
                };

                if w.add_particles(region.clone(), offset) {}
            }
        }


        //
        //Grab the state of the world
        //

        let particles = w.particles().particles();

        let n = w.particles().particles().len();
        let m = w.particles().boundary().len();
        let (den1, mat1,_, pos1, vel1) = Particle::get_attributes(&w.particles().boundary());
        let (den2, mat2,_, pos2, vel2) = Particle::get_attributes(&*particles);

        let (width, height) = self.window.borrow().get_framebuffer_size();
        let s = width.min(height);

        //
        //Draw to the framebuffer for recording
        //
        if self.pixels.is_some() {
            unsafe {
                const BUFFERS: [GLenum; 1] = [gl::COLOR_ATTACHMENT0];
                const CLEAR_COLOR: [GLfloat; 4] = [1.0,0.0,1.0,0.0];
                gl::BindFramebuffer(gl::FRAMEBUFFER, self.fb);
                gl::DrawBuffers(1, &BUFFERS[0] as *const GLenum);

                let s2 = self.rec_w.min(self.rec_h);
                gl::PointSize(s2 as f32 * *self.shader.render_h);
                gl::Viewport(
                    ((self.rec_w-s2)/2) as GLint, ((self.rec_h-s2)/2) as GLint,
                    s2 as GLint, s2 as GLint
                );

                gl::ClearBufferfi(gl::DEPTH_STENCIL, 0, 1.0, 0);
                gl::ClearBufferfv(gl::COLOR, 0, &CLEAR_COLOR[0] as *const GLfloat);

                gl::ClearColor(1.0,1.0,1.0,0.0);
                gl::ClearDepth(1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            }

            self.shader.draw(&mut self.context, DrawMode::Points, m, den1, mat1, pos1, pos1, vel1);
            self.shader.draw(&mut self.context, DrawMode::Points, n, den2, mat2, pos2, pos2, vel2);
        }

        //
        //Draw to the screen
        //
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::DrawBuffer(gl::BACK);
            gl::ClearColor(1.0,1.0,1.0,0.0);
            gl::ClearDepth(1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::PointSize(s as f32 * *self.shader.render_h);
            gl::Viewport((width-s)/2, (height-s)/2, s, s);
        }

        self.shader.draw(&mut self.context, DrawMode::Points, m, den1, mat1, pos1, pos1, vel1);
        self.shader.draw(&mut self.context, DrawMode::Points, n, den2, mat2, pos2, pos2, vel2);

        //
        //Update the window
        //
        glfw::Context::swap_buffers(&mut *self.window.borrow_mut());
        self.glfw.poll_events();
    }

}
