glsl!{$

    pub use self::poly6::*;

    pub(self) fn volume_nball(n: u32, r: f32) -> f32 {
        match n {
            0 => 1.0,
            1 => 2.0*r,
            _ => 2.0*::std::f32::consts::PI*r*r*volume_nball(n-2, r) / n as f32
        }
    }

    mod poly4 {

        @Rust
            use super::volume_nball;

            #[allow(dead_code)]
            pub fn norm_const(n: u32, h: f32) -> f32 {
                ((n+2)*(n+4)) as f32 / (8.0*h*h*h*h*volume_nball(n, h))
            }

        @Lib

            public float normalization_constant(int n, float h) {
                const float tau = 6.283185307;
                float v_n = 1;
                float i = float(n);
                for(; i>1; i-=2) {
                    v_n *= (tau*h*h)/i;
                }
                if(i==1){
                    v_n*= 2*h;
                }

                return float((n+2)*(n+4)) / (8.0*h*h*h*h*v_n);
            }

            public float kernel(vec4 r, float h, float k) {
                float diff = max(h*h - dot(r,r), 0);
                return k * diff * diff;
            }

            public vec4 grad_w(vec4 r, float h, float k) {
                return -2*k * max(h*h - dot(r,r), 0) * r;
            }

            public float lapl_w(vec4 r, float h, float k, uint n) {
                return 8*k * ((2-float(n))*dot(r,r) - float(n)*h*h);
            }

    }

    mod poly6 {

        @Rust
            use super::volume_nball;

            #[allow(dead_code)]
            pub fn norm_const(n: u32, h: f32) -> f32 {
                let pi = ::std::f32::consts::PI;
                (pi*pi*pi) / (6.0 * volume_nball(n+6, h))
            }

        @Lib

            public float normalization_constant(int n, float h) {
                const float tau = 6.283185307;
                float v_n6 = 1;
                float i = float(n+6);
                for(; i>1; i-=2) {
                    v_n6 *= (tau*h*h)/i;
                }
                if(i==1){
                    v_n6*= 2*h;
                }

                return (tau*tau*tau) / (48 * v_n6);
            }

            public float kernel(vec4 r, float h, float k) {
                float diff = max(h*h - dot(r,r), 0);
                return k * diff * diff * diff;
            }

            public vec4 grad_w(vec4 r, float h, float k) {
                float diff = max(h*h - dot(r,r), 0);
                return -6*k * diff * diff * r;
            }

            public float lapl_w(vec4 r, float h, float k, uint n) {
                float diff = max(h*h - dot(r,r), 0);
                return 24*k * diff * (dot(r,r) - (float(n)/4.0) * diff);
            }

    }

    mod lucy {
        @Rust
            use super::volume_nball;

            #[allow(dead_code)]
            pub fn norm_const(n: u32, h: f32) -> f32 {
                ((n+2)*(n+3)*(n+4)) as f32 / (24.0 * h*h*h*h* volume_nball(n, h))
            }

        @Lib

            public float normalization_constant(int n, float h) {
                const float tau = 6.283185307;
                float v_n = 1;
                float i = float(n);
                for(; i>1; i-=2) {
                    v_n *= (tau*h*h)/i;
                }
                if(i==1){
                    v_n*= 2*h;
                }

                return float((n+2)*(n+3)*(n+4)) / (24.0 * h*h*h*h* v_n);
            }

            public float kernel(vec4 r, float h, float k) {
                float len = length(r);
                float diff = max(h - len, 0);
                return k * (h + 3*len) * diff * diff * diff;
            }

            public vec4 grad_w(vec4 r, float h, float k) {
                float len = length(r);
                float diff = max(h - len, 0);
                return -12 * k * diff * diff * r;
            }

            public float lapl_w(vec4 r, float h, float k, uint n) {
                float len = length(r);
                return -12 * k * (h - len) * (n*h - (n+2)*len);
            }

    }

    mod spikey {
        @Rust
            use super::volume_nball;

            #[allow(dead_code)]
            pub fn norm_const(n: u32, h: f32) -> f32 {
                ((n+1)*(n+2)*(n+3)) as f32 / (6.0 * h*h*h* volume_nball(n, h))
            }

        @Lib

            public float normalization_constant(int n, float h) {
                const float tau = 6.283185307;
                float v_n = 1;
                float i = float(n);
                for(; i>1; i-=2) {
                    v_n *= (tau*h*h)/i;
                }
                if(i==1){
                    v_n*= 2*h;
                }

                return float((n+1)*(n+2)*(n+3)) / (6 *h*h*h* v_n);
            }

            public float kernel(vec4 r, float h, float k) {
                float diff = max(h - length(r), 0);
                return k * diff * diff * diff;
            }

            public vec4 grad_w(vec4 r, float h, float k) {
                float l = length(r);
                if(l==0) return vec4(0,0,0,0);
                float diff = max(h - l, 0);
                return (-3*k/l) * diff * diff * r;
            }

            public float lapl_w(vec4 r, float h, float k, uint n) {
                float l = length(r);
                if(l==0) return 0;
                float diff = max(h - l, 0);
                return (3*k/l) * diff * ((n+1)*l - (n-1)*h);
            }

    }
}
