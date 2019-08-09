
use super::*;

glsl!{$

    pub use self::glsl::*;

    mod glsl {
        @Lib

            public struct AABB {
                vec4 min;
                vec4 dim;
            };

            public struct AABE {
                vec4 center;
                vec4 radii;
            }

        @Rust

            use super::*;

            impl AABB {

                pub fn from_min_max(min:vec4, max:vec4) -> Self {
                    let mut dim = max;
                    for i in 0..4 {dim.value[i] -= min.value[i];}
                    AABB {min: min, dim: dim}
                }

                pub fn center(&self) -> vec4 {
                    [self.min[0]+0.5*self.dim[0],self.min[1]+0.5*self.dim[1],self.min[2]+0.5*self.dim[2],self.min[3]+0.5*self.dim[3]].into()
                }

                pub fn border(&self, b: f32) -> Difference<AABB, AABB> {
                    let (m, c, d) = (self.min, self.center(), self.dim);
                    Difference(
                        *self,
                        AABB {
                            min: [(m[0]+b).min(c[0]), (m[1]+b).min(c[1]), (m[2]+b).min(c[2]), (m[3]+b).min(c[3])].into(),
                            dim: [(d[0]-2.0*b).max(0.0), (d[1]-2.0*b).max(0.0), (d[2]-2.0*b).max(0.0), (d[3]-2.0*b).max(0.0)].into()
                        }
                    )
                }

                pub fn max(&self) -> vec4 {
                    [self.min[0]+self.dim[0],self.min[1]+self.dim[1],self.min[2]+self.dim[2],self.min[3]+self.dim[3]].into()
                }
            }

            impl AABE {
                pub fn border(&self, b: f32) -> Difference<AABE, AABE> {
                    let r = self.radii;
                    Difference(
                        *self,
                        AABE {
                            center: self.center,
                            radii: [(r[0]-b).max(0.0), (r[1]-b).max(0.0), (r[2]-b).max(0.0), (r[3]-b).max(0.0)].into()
                        }
                    )
                }
            }

            impl Region for AABB {
                #[inline] fn bound(&self) -> AABB { *self }
                fn contains(&self, p: vec4) -> bool {
                    for i in 0..4 {
                        if p[i] < self.min[i] || p[i]-self.min[i] > self.dim[i] {
                            return false;
                        }
                    }
                    return true;
                }
            }

            impl Region for AABE {
                #[inline]
                fn bound(&self) -> AABB {
                    let (c, a) = (self.center, self.radii);
                    AABB {
                        min: [c[0]-a[0], c[1]-a[1], c[2]-a[2], c[3]-a[3]].into(),
                        dim: [2.0*a[0], 2.0*a[1], 2.0*a[2], 2.0*a[3]].into()
                    }
                }

                fn contains(&self, p: vec4) -> bool {
                    let lhs = (0..4).fold(0.0, |c, i| {
                        let d = (p[i] - self.center[i]) as f64;
                        let r = self.radii[i] as f64;
                        if r > 0.0 {
                            let coeff = (0..4).fold(1.0, |prod, j| {
                                if self.radii[j]>0.0 && j!=i {
                                    prod * self.radii[i] as f64
                                } else {
                                    prod
                                }
                            });
                            c + d*d*coeff*coeff
                        } else {
                            c
                        }
                    });
                    let rhs = (0..4).fold(1.0, |rhs, i| if self.radii[i]>0.0 {rhs * self.radii[i] as f64} else {rhs});
                    lhs <= rhs * rhs
                }
            }

    }

}

pub trait Region: 'static {
    fn bound(&self) -> AABB;
    fn contains(&self, p: vec4) -> bool;
}

impl<R:Region+?Sized,Ptr: ::std::ops::Deref<Target=R> + 'static> Region for Ptr {
    fn bound(&self) -> AABB {self.deref().bound()}
    fn contains(&self, p: vec4) -> bool { self.deref().contains(p) }
}


#[derive(Clone,PartialEq)]
pub struct Transformed<R:Region>(pub R, pub mat4, pub vec4);

impl<R:Region> Transformed<R> {
    pub fn from_translation(region:R, trans:vec4) -> Self {
        Transformed(region, [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]].into(), trans)
    }
}

impl<R:Region> Region for Transformed<R> {
    fn bound(&self) -> AABB{
        let mut bound = self.0.bound();
        let mut min = self.2;
        let mut max = self.2;
        for i in 0..4 {
            let mut a_min = 0.0;
            for j in 0..4 {a_min += self.1.value[j][i]*bound.min.value[j];}

            min.value[i] += a_min;
            max.value[i] += a_min;

            for j in 0..4 {min.value[i] += (self.1.value[j][i]*bound.dim.value[j]).min(0.0);}
            for j in 0..4 {max.value[i] += (self.1.value[j][i]*bound.dim.value[j]).max(0.0);}

        }
        AABB::from_min_max(min, max)
    }

    fn contains(&self, p: vec4) -> bool {
        let mut p_prime = p.value;
        let mut a = self.1.value;

        for i in 0..4 {p_prime[i] -= self.2.value[i];}

        fn row_mul(a: &mut [[f32;4];4], v:&mut [f32;4], row:usize, col:usize, factor:f32) {
            for k in col..4 {a[k][row] *= factor;}
            v[row] *= factor;
        }

        fn row_sum(a: &mut [[f32;4];4], v:&mut [f32;4], row:usize, col:usize, factor:f32, dest:usize) {
            for k in col..4 {
                a[k][dest] += factor*a[k][row];
            }
            v[dest] += factor*v[row];
        }

        fn swap_rows(a: &mut [[f32;4];4], v:&mut [f32;4], row1:usize, row2:usize, col:usize) {
            for k in col..4 {
                a[k].swap(row1, row2);
            }
            v.swap(row1, row2);
        }

        //Guass-Jordon elimination
        let mut i = 0;
        let mut j = 0;
        while i<4 && j<4 {

            // println!("a={:?}, p={:?}", a, p_prime);

            if a[j][i] != 0.0 {
                let pivot = a[j][i];
                if pivot != 1.0 { row_mul(&mut a, &mut p_prime, i, j, 1.0/pivot); }
                for k in 0..4 {
                    if k!=i && a[j][k] != 0.0 {
                        let pivot2 = -a[j][k];
                        // row_mul(&mut a, &mut p_prime, k, j, pivot);
                        row_sum(&mut a, &mut p_prime, i, j, pivot2, k);
                    }
                }

                i += 1;
                j += 1;
            } else {
                let mut cont = false;
                for k in (i+1)..4 {
                    if a[j][k] != 0.0 {
                        swap_rows(&mut a, &mut p_prime, i, k, j);
                        cont = true;
                    }
                }
                if !cont {j += 1;}
            }
        }

        // let mut check = [0.0,0.0,0.0,0.0];
        // for i in 0..4 {
        //     for j in 0..4 {
        //         check[i] += self.1.value[j][i]*p_prime[j];
        //     }
        //     check[i] += self.2.value[i];
        // }
        //
        // println!("{:?}", a);
        // println!("{:?}", self.1.value);
        // println!("{:?}", p_prime);
        // println!("{:?} {:?}", p.value, check);

        self.0.contains(p_prime.into())
    }
}


#[derive(Clone,PartialEq,Eq)]
pub struct Difference<L:Region, R:Region>(pub L, pub R);
impl<L:Region, R:Region> Region for Difference<L, R> {
    fn bound(&self) -> AABB{self.0.bound()}
    fn contains(&self, p: vec4) -> bool {self.0.contains(p) && !self.1.contains(p)}
}

#[derive(Clone,PartialEq,Eq)]
pub struct Union<L:Region, R:Region>(pub L, pub R);
impl<L:Region, R:Region> Region for Union<L, R> {
    fn bound(&self) -> AABB{
        let (l,r) = (self.0.bound(), self.1.bound());
        let (m1, m2) = (l.max(), r.max());
        let min = [l.min[0].min(r.min[0]),l.min[1].min(r.min[1]),l.min[2].min(r.min[2]),l.min[3].min(r.min[3])];
        let max = [m1[0].max(m2[0]), m1[1].max(m2[1]), m1[2].max(m2[2]), m1[3].max(m2[3])];
        AABB{
            min: min.into(),
            dim: [max[0]-min[0],max[1]-min[1],max[2]-min[2],max[3]-min[3]].into(),
        }
    }
    fn contains(&self, p: vec4) -> bool {self.0.contains(p) || self.1.contains(p)}
}

#[derive(Clone,PartialEq,Eq)]
pub struct Intersection<L:Region, R:Region>(pub L, pub R);
impl<L:Region, R:Region> Region for Intersection<L, R> {
    fn bound(&self) -> AABB{
        let (l,r) = (self.0.bound(), self.1.bound());
        let (max1, max2) = (l.max(), r.max());

        let mut intersection = AABB{
            min: [0.0,0.0,0.0,0.0].into(),
            dim: [0.0,0.0,0.0,0.0].into(),
        };

        for i in 0..4 {
            intersection.min[i] = l.min[i].max(r.min[i]);
            intersection.dim[i] = max1[i].min(max2[i]) - intersection.min[i];
        }

        intersection

    }
    fn contains(&self, p: vec4) -> bool {self.0.contains(p) && self.1.contains(p)}
}
