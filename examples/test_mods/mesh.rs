
use super::*;

pub struct Mesh {
    pub mesh:IndexedMesh
}

pub struct Surface {
    pub mesh:IndexedMesh,
    pub border: f32
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
