import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import meshio
from collections import defaultdict
import pyvista as pv
import time
from datetime import timedelta
from tqdm import tqdm
from scipy.sparse.linalg import gmres

import taichi as ti
import taichi.math as tm

from pathlib import Path
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = Path(f"results/simulation_{timestamp}")
output_folder.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cpu, default_fp=ti.f64)

vec2 = ti.types.vector(2, ti.f64)
vec3 = ti.types.vector(3, ti.f64)
vec3_int = ti.types.vector(3, ti.i32)

start_time = time.perf_counter()
c = 3e8
# f = 20e9
# wavelength = c/f
wavelength = 1
f = c/wavelength
k = 2*np.pi/wavelength
w = 2*np.pi*f
u = 4*np.pi*1e-7
epsilon = 8.85*1e-12

# mesh = meshio.read(r"meshes/plane50points.msh")
mesh = meshio.read(r"meshes/MoM_Test_Plane-TestPlane.msh")
# mesh = meshio.read(r"meshes/plane45pointsFDQ_experimental.msh")
# mesh = meshio.read(r"meshes/plane35pointsDelaunay.msh")
# mesh = meshio.read(r"meshes/plane35pointsDelaunay.msh")
# mesh = meshio.read(r"meshes/sphere40points.msh")
# mesh = meshio.read(r"meshes/parabola.msh")
nop = ti.static(7)
configuration_name = "Plane_50Points_nop7"

points_np = mesh.points.astype(np.float64)
triangles_np = mesh.cells_dict["triangle"].astype(np.int32)

print("Triangles:", len(triangles_np))

for i in range(0,len(points_np)):
    points_np[i] = (np.array(points_np[i])/1e6)
for i in range(0,len(triangles_np)):
    triangles_np[i] = np.array(triangles_np[i])

min_corner = np.min(points_np, axis=0)
max_corner = np.max(points_np, axis=0)
print(min_corner)
print(max_corner)
bbox_center = (min_corner + max_corner) / 2.0
points_np -= bbox_center

avg_side_length = 0
for tri in points_np[triangles_np]:
    s1 = np.linalg.norm(tri[0] - tri[1])
    s2 = np.linalg.norm(tri[0] - tri[2])
    s3 = np.linalg.norm(tri[2] - tri[1])
    avg_side_length += (s1 + s2 + s3) / 3

avg_side_length = avg_side_length / len(triangles_np)

avg_side_length_ti = ti.field(dtype=ti.f64, shape=())
avg_side_length_ti[None] = avg_side_length

n_points = points_np.shape[0]
points_ti = ti.Vector.field(3, dtype=ti.f64, shape=n_points)

n_tri = triangles_np.shape[0]
triangles_ti = ti.Vector.field(3, dtype=ti.i32, shape=n_tri)

for i in range(n_points):
    points_ti[i] = points_np[i]  

for i in range(n_tri):
    triangles_ti[i] = triangles_np[i]   


############## 7 point
if nop == 7:
    alpha = ti.Vector([0.3333, 0.0597, 0.4701, 0.4701, 0.7974, 0.1013, 0.1013])
    beta  = ti.Vector([0.3333, 0.4701, 0.0597, 0.4701, 0.1013, 0.7974, 0.1013])
    gamma = ti.Vector([0.3333, 0.4701, 0.4701, 0.0597, 0.1013, 0.1013, 0.7974])
    w_gauss = ti.Vector([0.225, 0.1323, 0.1323, 0.1323, 0.1259, 0.1259, 0.1259])
############## 4 point
elif nop == 4:
    alpha = ti.Vector([0.3333333, 0.6000000, 0.2000000, 0.2000000])
    beta  = ti.Vector([0.3333333, 0.2000000, 0.6000000, 0.2000000])
    gamma = ti.Vector([0.3333333, 0.2000000, 0.2000000, 0.6000000])
    w_gauss     = ti.Vector([-0.56250000, 0.52083333, 0.52083333, 0.52083333])
############# 6 point
elif nop == 6:
    alpha = ti.Vector([0.10810301, 0.44594849, 0.44594849, 0.81684757, 0.09157621, 0.09157621])
    beta  = ti.Vector([0.44594849, 0.10810301, 0.44594849, 0.09157621, 0.81684757, 0.09157621])
    gamma = ti.Vector([0.44594849, 0.44594849, 0.10810301, 0.09157621, 0.09157621, 0.81684757])
    w_gauss     = ti.Vector([0.22338158, 0.22338158, 0.22338158, 0.10995174, 0.10995174, 0.10995174])
############# 3 point
elif nop == 3:
    alpha = ti.Vector([0.66666667, 0.16666667, 0.16666667])
    beta  = ti.Vector([0.16666667, 0.66666667, 0.16666667])
    gamma = ti.Vector([0.16666667, 0.16666667, 0.66666667])
    w_gauss     = ti.Vector([0.33333333, 0.33333333, 0.33333333])
############# 12 point
elif nop == 12:
    alpha = ti.Vector([0.249286745170910, 0.249286745170910, 0.501426509658179,
                    0.063089014491502, 0.063089014491502, 0.873821971016996,
                    0.310352451033785, 0.636502499121399, 0.053145049844816,
                    0.310352451033785, 0.636502499121399, 0.053145049844816])

    beta  = ti.Vector([0.249286745170910, 0.501426509658179, 0.249286745170910,
                    0.063089014491502, 0.873821971016996, 0.063089014491502,
                    0.636502499121399, 0.053145049844816, 0.310352451033785,
                    0.053145049844816, 0.310352451033785, 0.636502499121399])

    gamma = 1.0 - alpha - beta

    w_gauss     = ti.Vector([0.116786275726379, 0.116786275726379, 0.116786275726379,
                    0.050844906370207, 0.050844906370207, 0.050844906370207,
                    0.082851075618374, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.082851075618374, 0.082851075618374])

elif nop == 16:
    # Symmetric 16-point rule (Weights sum to 1.0)
    w_gauss = ti.Vector([
        0.14431560767778,                                     
        0.09509163426728, 0.09509163426728, 0.09509163426728, 
        0.10321735053038, 0.10321735053038, 0.10321735053038, 
        0.03245846762319, 0.03245846762319, 0.03245846762319, 
        0.02723063468594, 0.02723063468594, 0.02723063468594, 
        0.02723063468594, 0.02723063468594, 0.02723063468594  
    ])
    
    # Coordinates (Barycentric)
    alpha = ti.Vector([
        0.33333333333333,
        0.08308905341457, 0.45845547329272, 0.45845547329272,
        0.17056930775176, 0.41471534612412, 0.41471534612412,
        0.01323171575661, 0.49338414212170, 0.49338414212170,
        0.04615406080279, 0.75881140104859, 0.19503453814862,
        0.75881140104859, 0.19503453814862, 0.04615406080279
    ])
    
    beta = ti.Vector([
        0.33333333333333,
        0.45845547329272, 0.08308905341457, 0.45845547329272,
        0.41471534612412, 0.17056930775176, 0.41471534612412,
        0.49338414212170, 0.01323171575661, 0.49338414212170,
        0.19503453814862, 0.04615406080279, 0.75881140104859,
        0.19503453814862, 0.75881140104859, 0.04615406080279
    ])
    
    gamma = 1.0 - alpha - beta

ncap_global = []

for i, tri in enumerate(triangles_np):
    v = points_np[tri]
    normal = np.cross(v[1]-v[0], v[2]-v[0])
    centroid = (v[0] + v[1] + v[2])/3
    if np.dot(normal, centroid) < 0:
        triangles_np[i] = triangles_np[i][::-1]  # reverse winding

    v = points_np[triangles_np[i]]
    normal = np.cross(v[1]-v[0], v[2]-v[0])
    normal /= np.linalg.norm(normal)
    ncap_global.append(normal)

ncap_global = np.array(ncap_global)
x_ncap_len, y_ncap_len = ncap_global.shape
ncap_global_ti = ti.field(dtype=ti.f64, shape=(x_ncap_len, y_ncap_len))

for i in range(x_ncap_len):
    for j in range(y_ncap_len):
        ncap_global_ti[i, j] = ncap_global[i, j]

n_tri = len(triangles_np)

def get_area(triangle_np):
    tri = points_np[triangle_np]
    side1 = tri[1] - tri[0]
    side2 = tri[2] - tri[0]
    area = 0.5 * np.linalg.norm(np.cross(side1, side2))
    centroid = (tri[0] + tri[1] + tri[2])/3
    return area, centroid

triangle_area = np.zeros((len(triangles_np)))
triangle_centroid = np.zeros((len(triangles_np), 3))

for i in range(0,len(triangles_np)):
    triangle_area[i], triangle_centroid[i] = get_area(triangles_np[i])

tri_prop = ti.types.struct(centroid = vec3,
                     area = ti.f64)
triangle_properties = tri_prop.field(shape=(len(triangles_np),))

for i in range(len(triangles_np)):
    triangle_properties[i] = tri_prop(
        centroid = ti.math.vec3(*triangle_centroid[i]),
        area = triangle_area[i]
    )

complex_vector = ti.Vector.field(n=2, dtype=ti.f32, shape=(3,))
singularity_I1I2 = ti.types.struct(I1 = vec3,
                     I2 = ti.f64)

@ti.func
def c_exp_j(theta: float) -> vec2:
    return vec2(ti.cos(theta), ti.sin(theta))


@ti.func
def func(rp: vec3, rq: vec3, 
         rp_type: ti.i32, rq_type: ti.i32, 
         rp_free: vec3, rq_free: vec3) -> vec2:
    distance = tm.length(rp - rq)

    sign_rp = 1 - 2 * rp_type
    sign_rq = 1 - 2 * rq_type

    rho_rp = sign_rp * (rp - rp_free)
    rho_rq = sign_rq * (rq - rq_free)

    j = vec2(0.0, 1.0)

    first_term = j * w * u * 0.25 * tm.dot(rho_rp, rho_rq)
    second_term = j / (w * epsilon) * sign_rp * sign_rq

    phase = c_exp_j(-k * distance)

    result = tm.cmul((first_term - second_term),phase) / distance

    return result


@ti.func
def func_singularity(rp: vec3, rq: vec3, rp_type: ti.i32, rq_type: ti.i32, 
                     rp_free: vec3, rq_free: vec3, tout: ti.i32, tin: ti.i32, 
                     inner: ti.i32) -> vec2:
    sign_rp = 1 - 2 * rp_type
    sign_rq = 1 - 2 * rq_type
    rho_rp = sign_rp * (rp - rp_free)
    rho_rq = sign_rq * (rq - rq_free)
    final_result = vec2(0, 0)
    if inner:
        # INNER TERM: [(e^(-jkR) - 1)/R] for singularity extraction
        # distance = tm.length(rp - rq)
        # eps = 1e-12
        
        # First part: [jωμ/4 ρ̂_m · ρ̂_n ± j/(ωε)]
        first_term = vec2(0, w*u/4) * tm.dot(rho_rp, rho_rq) - vec2(0, 1/(w*epsilon)) * sign_rp * sign_rq
        
        # Second part: [(e^(-jkR) - 1)/R]
        # if distance < eps:
        #     # R = 0: use limit = -jk
        #     first_term = tm.cmul(first_term, vec2(0, -k))
        # else:
        #     # R ≠ 0: use actual expression
        #     first_term = tm.cmul(first_term, (c_exp_j(-k * distance) - vec2(1, 0)) / distance)
        first_term = tm.cmul(first_term, vec2(0, -k))
        final_result = first_term
        
    else:
        tri0 = points_ti[triangles_ti[tin][0]]
        tri1 = points_ti[triangles_ti[tin][1]]
        tri2 = points_ti[triangles_ti[tin][2]]
        side11 = tri0 - tri1
        side12 = tri0 - tri2
        ncap = tm.cross(side11, side12)
        ncap /= tm.length(ncap)
        
        I1_I2_result = get_I1I2(tin, rp, ncap)
        I1 = I1_I2_result.I1
        I2 = I1_I2_result.I2

        inner_integral = sign_rq * (I1 + (rp - rq_free) * I2)
        second_term_scalar = tm.dot(rho_rp, inner_integral)
        second_term = vec2(0, w * u / 4) * second_term_scalar

        third_term = vec2(0, 1 / (w * epsilon)) * sign_rp * sign_rq * I2
        
        final_result = (second_term - third_term) / triangle_properties[tin].area
    
    return final_result

@ti.func
def get_component_in_plane(ncap: vec3, r:vec3) -> vec3:
    return r - ncap * tm.dot(ncap, r)

@ti.func
def get_I1I2(tin: ti.i32, obv_point: vec3, ncap: vec3) -> singularity_I1I2:
    eps = 1e-12
    I1 = vec3(0.0, 0.0, 0.0)
    I2 = 0.0
    p1 = vec3(0.0, 0.0, 0.0)
    p2 = vec3(0.0, 0.0, 0.0)
    
    for i in ti.static(range(3)):
        p1 = points_ti[triangles_ti[tin][i]]
        p2 = points_ti[triangles_ti[tin][(i + 1) % 3]]
            
        lcap = (p2 - p1)
        lcap = lcap / tm.length(lcap)

        ucap = tm.cross(lcap, ncap)

        obv_point_p = obv_point #+ ucap * eps

        rho = get_component_in_plane(ncap, obv_point_p)
        rho_p1 = get_component_in_plane(ncap, p1)
        rho_p2 = get_component_in_plane(ncap, p2)

        l_plus = tm.dot((rho_p2 - rho), lcap)
        l_minus = tm.dot((rho_p1 - rho), lcap)

        p0 = ti.abs(tm.dot(ucap, rho_p1 - rho))

        p_plus = tm.length(rho_p2 - rho)
        p_minus = tm.length(rho_p1 - rho)

        p0_cap = ((rho_p1 - rho) - l_minus * lcap)
        p0_cap /= (p0 + eps)  

        d = ti.abs(tm.dot(ncap, obv_point_p - p1))
        R0 = tm.sqrt(p0*p0 + d*d)
        Rplus = tm.sqrt(p_plus*p_plus + d*d)
        Rminus = tm.sqrt(p_minus*p_minus + d*d)
        

        common_term = tm.log((Rplus + l_plus + eps) / (Rminus + l_minus + eps))
        

        I1_value = (R0*R0) * common_term + l_plus*Rplus - l_minus*Rminus
        I1 += I1_value * ucap
        
        p0_cap_dot_u = tm.dot(p0_cap, ucap)

        tan_term1 = tm.atan2((p0 * l_plus) , (R0*R0 + d*Rplus + eps))
        tan_term2 = tm.atan2((p0 * l_minus) , (R0*R0 + d*Rminus + eps))

        log_term = p0 * common_term
        atan_term = d * (tan_term1 - tan_term2)
        I2 += p0_cap_dot_u * (log_term - atan_term)
    
    I1 = I1 / 2.0
    
    result = singularity_I1I2(I1=I1, I2=I2)
    return result


@ti.func
def get_triangle_relationship(tin: ti.i32, tout: ti.i32) -> ti.i32:

    tri_in = triangles_ti[tin]
    tri_out = triangles_ti[tout]
    shared = 0
    
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            if tri_in[i] == tri_out[j]:
                shared += 1
    
    return shared

@ti.func
def double_integration(vout1: vec3, vout2: vec3, vout3: vec3, 
                       vin1: vec3, vin2: vec3, vin3: vec3, 
                       rp_type: ti.i32, rq_type: ti.i32, 
                       rp_free: vec3, rq_free: vec3, 
                       tout: ti.i32, tin: ti.i32) -> vec2:
    final_result = vec2(0.0, 0.0)
    delta_rc = triangle_properties[tin].centroid - triangle_properties[tout].centroid
    delta_rc_length = tm.length(delta_rc)   
    if delta_rc_length > 0.1*avg_side_length_ti[None]:
    # shared = get_triangle_relationship(tin, tout)
    # if shared < 2:
        for i in range(nop):
            location_out = alpha[i]*vout1 + beta[i]*vout2 + gamma[i]*vout3
            result = vec2(0.0, 0.0)
            for j in range(nop):
                location_in = alpha[j]*vin1 + beta[j]*vin2 + gamma[j]*vin3
                result += func(location_out, location_in, rp_type, rq_type, rp_free, rq_free) * w_gauss[j]
            final_result += result * w_gauss[i]
    else:
        for i in range(nop): 
            location_out = alpha[i]*vout1 + beta[i]*vout2 + gamma[i]*vout3
            result = vec2(0.0, 0.0)          
            for j in range(nop): 
                location_in = alpha[j]*vin1 + beta[j]*vin2 + gamma[j]*vin3
                result += func_singularity(location_out, location_in, rp_type, rq_type, 
                                          rp_free, rq_free, tout, tin, 1) * w_gauss[j]
            result += func_singularity(location_out, location_out, rp_type, rq_type, 
                                      rp_free, rq_free, tout, tin, 0)
           
            final_result += result * w_gauss[i]   
    return final_result

@ti.func
def electric_field(r: vec3) -> ti.types.matrix(3, 2, float):
    phase = c_exp_j(-k * r[2])
    result = ti.Matrix([
        [1.0 * phase[0], 1.0 * phase[1]],  
        [0.0, 0.0],                         
        [0.0, 0.0]                          
    ])
    # result = ti.Matrix([
    #     [0.0, 0.0],  
    #     [1.0 * phase[0], 1.0 * phase[1]],                         
    #     [0.0, 0.0]                          
    # ])
    return result

@ti.func
def complex_dot(v_real: vec3, v_complex) -> vec2:
    result = vec2(0.0, 0.0)
    for i in range(3):
        result[0] += v_real[i] * v_complex[i, 0] 
        result[1] += v_real[i] * v_complex[i, 1] 
    return result

@ti.func
def get_excitation(vminus: vec3, vplus: vec3, common1: vec3, common2: vec3, edge_length: ti.f64) -> vec2:
    sum_minus = vec2(0.0, 0.0)
    sum_plus = vec2(0.0, 0.0)
    for i in range(nop):
        location_minus = alpha[i]*vminus + beta[i]*common1 + gamma[i]*common2
        rho_minus = location_minus - vminus
        func_minus = complex_dot(rho_minus, electric_field(location_minus))
        sum_minus += func_minus * w_gauss[i]
        
        location_plus = alpha[i]*vplus + beta[i]*common1 + gamma[i]*common2
        rho_plus = vplus - location_plus
        func_plus = complex_dot(rho_plus, electric_field(location_plus))
        sum_plus += func_plus * w_gauss[i]
    
    return (edge_length/2) * (sum_plus + sum_minus)

edge_dict = defaultdict(list)

for tri_id, tri in enumerate(triangles_np):
    edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
    for e in edges:
        e_sorted = tuple(sorted(e))
        edge_dict[e_sorted].append(tri_id)

shared_edges = {e: t for e, t in edge_dict.items() if len(t) == 2}
boundary_edges = {e: t for e, t in edge_dict.items() if len(t) == 1}

count = 0
all_basis = {}
organized_triangles = {}
for edge, tris in shared_edges.items():
    n1, n2 = edge
    t1, t2 = tris
    if t1 not in organized_triangles:
        organized_triangles[t1] = np.array([0,0,0], dtype=complex)

    if t2 not in organized_triangles:
        organized_triangles[t2] = np.array([0,0,0], dtype=complex)

    tri1 = triangles_np[t1]
    tri2 = triangles_np[t2]
    tri1_pts = points_np[tri1]
    tri2_pts = points_np[tri2]
    side11 = tri1_pts[0] - tri1_pts[1]
    side12 = tri1_pts[0] - tri1_pts[2]
    area1 = 0.5 * np.linalg.norm(np.cross(side11, side12))
    side21 = tri2_pts[0] - tri2_pts[1]
    side22 = tri2_pts[0] - tri2_pts[2]
    area2 = 0.5 * np.linalg.norm(np.cross(side21, side22))
    edge_length = np.linalg.norm(points_np[n2] - points_np[n1])
    pos_tri, neg_tri = t1, t2
    pos_area, neg_area = area1, area2
    free_pos = points_np[list(set(triangles_np[pos_tri]) - set(edge))[0]]
    free_neg = points_np[list(set(triangles_np[neg_tri]) - set(edge))[0]]
    common1 = points_np[n1]
    common2 = points_np[n2]
    common_points = [common1, common2]
    all_basis[count] = [free_neg, free_pos, edge_length, common_points, pos_tri, neg_tri, pos_area, neg_area]
    ######################  0        1           2              3          4        5          6       7
    count = count + 1

s1 = ti.types.struct(free_neg = vec3, 
                     free_pos = vec3, 
                     edge_length = ti.f64, 
                     cp1 = vec3, 
                     cp2 = vec3, 
                     neg_tri = ti.i32, 
                     pos_tri = ti.i32, 
                     pos_area = ti.f64, 
                     neg_area = ti.f64)
all_basis_ti = s1.field(shape=(count,))

for i in range(count):
    all_basis_ti[i] = s1(
        free_neg=ti.math.vec3(*all_basis[i][0]), 
        free_pos=ti.math.vec3(*all_basis[i][1]),
        edge_length=all_basis[i][2],
        cp1=ti.math.vec3(*all_basis[i][3][0]),
        cp2=ti.math.vec3(*all_basis[i][3][1]),
        neg_tri=all_basis[i][5],
        pos_tri=all_basis[i][4],
        pos_area=all_basis[i][6],
        neg_area=all_basis[i][7]
    )

print(all_basis_ti.shape[0])


N = all_basis_ti.shape[0]
Z = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(N, N)) 
I = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(N,))   

@ti.kernel
def get_Z():
    for m in range(N):
        mth_basis = all_basis_ti[m]
        I[m] = get_excitation(mth_basis.free_neg, mth_basis.free_pos, 
                             mth_basis.cp1, mth_basis.cp2, mth_basis.edge_length)
        
        for n in range(N):
            nth_basis = all_basis_ti[n]
            
            Imneg_nneg = double_integration(
                mth_basis.free_neg, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_neg, nth_basis.cp1, nth_basis.cp2, 
                0, 0, mth_basis.free_neg, nth_basis.free_neg, 
                mth_basis.neg_tri, nth_basis.neg_tri
            )
            
            Imneg_npos = double_integration(
                mth_basis.free_neg, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_pos, nth_basis.cp1, nth_basis.cp2, 
                0, 1, mth_basis.free_neg, nth_basis.free_pos, 
                mth_basis.neg_tri, nth_basis.pos_tri
            )
            
            Impos_nneg = double_integration(
                mth_basis.free_pos, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_neg, nth_basis.cp1, nth_basis.cp2, 
                1, 0, mth_basis.free_pos, nth_basis.free_neg, 
                mth_basis.pos_tri, nth_basis.neg_tri
            )
            
            Impos_npos = double_integration(
                mth_basis.free_pos, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_pos, nth_basis.cp1, nth_basis.cp2, 
                1, 1, mth_basis.free_pos, nth_basis.free_pos, 
                mth_basis.pos_tri, nth_basis.pos_tri
            )
            
            factor = (mth_basis.edge_length * nth_basis.edge_length) / (4 * ti.math.pi)
            Z[m, n] = factor * (Imneg_nneg + Imneg_npos + Impos_nneg + Impos_npos)

get_Z()

Z_np = Z.to_numpy()
I_np = I.to_numpy()

Z_complex = Z_np[:, :, 0] + 1j * Z_np[:, :, 1]
I_complex = I_np[:, 0] + 1j * I_np[:, 1]

Z_complex = np.round(Z_complex, 12)
I_complex = np.round(I_complex, 12)
coeff = np.linalg.solve(Z_complex, I_complex)
print("Coefficients Obtained")
# cond = np.linalg.cond(Z_complex)
# print(f"Condition Number: {cond}")
Z_file_name = configuration_name + "_Z.npy"
I_file_name = configuration_name + "_I.npy"
coeff_file_name = configuration_name + "_coeff.npy"
# np.save(output_folder/Z_file_name, Z_complex)
# np.save(output_folder/I_file_name, I_complex)
# np.save(output_folder/coeff_file_name, coeff)

# print("Solving with GMRES...")
# coeff, info = gmres(
#     Z_complex, 
#     I_complex
# )

# if info == 0:
#     print("✓ GMRES converged successfully")
#     residual = np.linalg.norm(Z_complex @ coeff - I_complex)
#     print(f"Residual: {residual:.6e}")
# else:
#     print(f"✗ GMRES did not converge (info={info})")

coeff_ti = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(N,))
for i in range(N):
    coeff_ti[i] = vec2(coeff[i].real, coeff[i].imag)

for i in tqdm(range(N), desc="Assembling colors"):
    ith_basis = all_basis[i]
    value = coeff[i]*ith_basis[2]/2

    pos_tri = points_np[triangles_np[ith_basis[4]]]
    pos_centroid = (pos_tri[0] + pos_tri[1] + pos_tri[2])/3
    pos_vec = ith_basis[1] - pos_centroid
    organized_triangles[ith_basis[4]] += (value/ith_basis[6])*pos_vec

    neg_tri = points_np[triangles_np[ith_basis[5]]]
    neg_centroid = (neg_tri[0] + neg_tri[1] + neg_tri[2])/3
    neg_vec = neg_centroid - ith_basis[0]
    organized_triangles[ith_basis[5]] += (value/ith_basis[7])*neg_vec


for key in organized_triangles:
    organized_triangles[key] = np.linalg.norm(organized_triangles[key])

max_current = max(organized_triangles.values())
for key in organized_triangles:
    organized_triangles[key] = organized_triangles[key]/max_current

print("Done getting organized_triangles")

vertices = []
faces = []
scalars = []

for tri_idx, value in organized_triangles.items():
    tri = points_np[triangles_np[tri_idx]]
    start_index = len(vertices)
    vertices.extend(tri)
    faces.extend([3, start_index, start_index + 1, start_index + 2])
    scalars.append(np.linalg.norm(value))
vertices = np.array(vertices)
mesh = pv.PolyData(vertices, faces)
mesh.cell_data["Magnitude"] = np.array(scalars)

mag = np.array(scalars)
vmin, vmax = 0, mag.max()
mesh_file_name = configuration_name + "_mesh.vtk"
mesh.save(output_folder/mesh_file_name)
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="Magnitude", cmap="jet", show_edges=False)
plotter.add_scalar_bar(title="|Current| (A/m)")
plotter.show_axes()
plotter.show_bounds(grid='front', location='outer', all_edges=True)


@ti.func
def realVec_mul_complexScalar(realVec: vec3, complexScalar: vec2) -> ti.types.matrix(3, 2, ti.f64):
    result = ti.Matrix([
        [realVec[0]*complexScalar[0], realVec[0]*complexScalar[1]],  
        [realVec[1]*complexScalar[0], realVec[1]*complexScalar[1]],
        [realVec[2]*complexScalar[0], realVec[2]*complexScalar[1]]
    ])
    return result

@ti.func
def complexVec_mul_complexScalar(complexVec: ti.types.matrix(3, 2, ti.f64), complexScalar: vec2) -> ti.types.matrix(3, 2, ti.f64):
    x_comp = vec2(complexVec[0,0],complexVec[0,1])
    y_comp = vec2(complexVec[1,0],complexVec[1,1])
    z_comp = vec2(complexVec[2,0],complexVec[2,1])
    x_prod = tm.cmul(x_comp, complexScalar)
    y_prod = tm.cmul(y_comp, complexScalar)
    z_prod = tm.cmul(z_comp, complexScalar)
    result = ti.Matrix([
        [x_prod[0], x_prod[1]],  
        [y_prod[0], y_prod[1]],  
        [z_prod[0], z_prod[1]]   
    ])
    return result

@ti.func
def far_field(rhat: vec3, m: ti.i32, r: ti.f64):
    mth_basis = all_basis_ti[m]
    
    result_neg = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result_pos = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    
    I_m = coeff_ti[m]

    for i in range(nop):
        location_neg = alpha[i]*mth_basis.free_neg + beta[i]*mth_basis.cp1 + gamma[i]*mth_basis.cp2
        rho_neg = location_neg - mth_basis.free_neg
        phase_neg = c_exp_j(k * tm.dot(rhat, location_neg))
        result_neg += realVec_mul_complexScalar(w_gauss[i] * rho_neg, phase_neg)

        location_pos = alpha[i]*mth_basis.free_pos + beta[i]*mth_basis.cp1 + gamma[i]*mth_basis.cp2
        rho_pos = mth_basis.free_pos - location_pos
        phase_pos = c_exp_j(k * tm.dot(rhat, location_pos))
        result_pos += realVec_mul_complexScalar(w_gauss[i] * rho_pos, phase_pos)
    
    integral_neg = complexVec_mul_complexScalar(result_neg, vec2(mth_basis.neg_area, 0.0))
    integral_pos = complexVec_mul_complexScalar(result_pos, vec2(mth_basis.pos_area, 0.0))
    combined_integral = complexVec_mul_complexScalar(integral_neg + integral_pos, I_m)
    
    const_factor = tm.cmul(vec2(0.0, -w * u), c_exp_j(-k * r)) * (mth_basis.edge_length / (8.0 * tm.pi * r))
    
    return complexVec_mul_complexScalar(combined_integral, const_factor)

print("DONE GETTING REFERENCE")
#############################################################################################
wavenumber = 2*np.pi/wavelength

mesh = meshio.read(r"surface_D.msh")
reconstructed_coordinates = mesh.points.astype(np.float64)
reconstructed_quad = mesh.cells_dict["quad"].astype(np.int32)
reconstructed_coordinates = reconstructed_coordinates
for i in range(len(reconstructed_quad)):
    quad = reconstructed_quad[i]
    pts = reconstructed_coordinates[quad]
    
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)  
    
    reconstructed_quad[i] = quad[sorted_indices]

centroids = reconstructed_coordinates[reconstructed_quad[:,0]]#.mean(axis=1)  # (N, 3)
cx = np.round(centroids[:, 0], decimals=6)
cy = np.round(centroids[:, 1], decimals=6)

sorted_indices = np.lexsort((cx, cy))
reconstructed_quad = reconstructed_quad[sorted_indices]

n_rows = int(np.sqrt(reconstructed_quad.shape[0]))
n_cols = n_rows

print(n_rows, n_cols)

reconstructed_quad_2d = reconstructed_quad.reshape(n_rows, n_cols, 4)

p0 = reconstructed_coordinates[reconstructed_quad[0][0]]
p1 = reconstructed_coordinates[reconstructed_quad[0][1]]
del_x = np.linalg.norm(p0-p1)

p2 = reconstructed_coordinates[reconstructed_quad[0][2]]
del_y = np.linalg.norm(p2-p1)

reconstructed_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(reconstructed_coordinates)))
reconstructed_quad_ti = ti.Vector.field(n=4, dtype=ti.int32, shape=(len(reconstructed_quad)))

for i in range(0,len(reconstructed_coordinates)):
    reconstructed_coordinates_ti[i] = reconstructed_coordinates[i]

for i in range(0,len(reconstructed_quad)):
    reconstructed_quad_ti[i] = reconstructed_quad[i]

mesh2 = meshio.read(r"surface_s1_p.msh")
reconstructed_coordinates2 = mesh2.points.astype(np.float64)
reconstructed_quad2 = mesh2.cells_dict["quad"].astype(np.int32)

for i in range(len(reconstructed_quad2)):
    quad = reconstructed_quad2[i]
    pts = reconstructed_coordinates2[quad]  # shape (4, 3)
    
    centroid2 = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid2[1], pts[:, 0] - centroid2[0])
    sorted_indices = np.argsort(angles)  # counter-clockwise order
    
    reconstructed_quad2[i] = quad[sorted_indices]

centroids2 = reconstructed_coordinates2[reconstructed_quad2[:,0]]#.mean(axis=1)  # (N, 3)
cx = np.round(centroids2[:, 0], decimals=6)
cy = np.round(centroids2[:, 1], decimals=6)

sorted_indices = np.lexsort((cx, cy))
reconstructed_quad2 = reconstructed_quad2[sorted_indices]

###################################################### chaging coordinates measurement plane
surface1_coordinates = reconstructed_coordinates2 + [0,0,5]
surface2_coordinates = reconstructed_coordinates2 + [0,0,7]

surface1_quad = mesh2.cells_dict["quad"].astype(np.int32)
surface2_quad = mesh2.cells_dict["quad"].astype(np.int32)

surface1_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(surface1_coordinates)))
surface1_quad_ti = ti.Vector.field(n=4, dtype=ti.int32, shape=(len(surface1_quad)))

for i in range(0,len(surface1_coordinates)):
    surface1_coordinates_ti[i] = surface1_coordinates[i]

for i in range(0,len(surface1_quad)):
    surface1_quad_ti[i] = surface1_quad[i]

surface2_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(surface2_coordinates)))
surface2_quad_ti = ti.Vector.field(n=4, dtype=ti.int32, shape=(len(surface2_quad)))

for i in range(0,len(surface2_coordinates)):
    surface2_coordinates_ti[i] = surface2_coordinates[i]

for i in range(0,len(surface2_quad)):
    surface2_quad_ti[i] = surface2_quad[i]

obs_plane1 = np.int32(len(surface1_coordinates))
obs_plane2 = np.int32(len(surface2_coordinates))

N_quad = len(reconstructed_quad)
N1_quad = len(surface1_quad)
N2_quad = len(surface2_quad)

N_coordinates = len(reconstructed_coordinates)
N1_coordinates = len(surface1_coordinates)
N2_coordinates = len(surface2_coordinates)
# theta_np = np.linspace(-np.pi, np.pi, num=observations)
# r = 1.0
# phi = 0.0
@ti.func
def green_func(r: vec3, rprime: vec3) -> vec2:
    distance = tm.length(r - rprime)
    theta = -wavenumber*distance
    comp_vec = vec2(ti.cos(theta), ti.sin(theta))/(4*ti.math.pi*distance)
    return comp_vec

@ti.func
def green_func_derivative(r: vec3, rprime: vec3, wrt: ti.int32) -> vec2: # wrt --> 0:x, 1:y, 2:z
    distance = tm.length(r - rprime)
    theta = -wavenumber*distance
    comp_vec = vec2(ti.cos(theta), ti.sin(theta))/(4*ti.math.pi*(distance**3))
    comp_vec2 = vec2(1,wavenumber*distance)*(r[wrt] - rprime[wrt])
    result = tm.cmul(comp_vec,comp_vec2)
    return result

f1 = ti.Vector.field(n=2, dtype=ti.f64, shape=(2*N1_coordinates,))
f2 = ti.Vector.field(n=2, dtype=ti.f64, shape=(2*N2_coordinates,))


E_field_s1 = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(obs_plane1, 3))  # Complex E-field vectors
E_magnitude_s1 = ti.field(dtype=ti.f64, shape=(obs_plane1,))

E_field_s2 = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(obs_plane2, 3))  # Complex E-field vectors
E_magnitude_s2 = ti.field(dtype=ti.f64, shape=(obs_plane2,))

@ti.kernel
def compute_f1():
    """Compute far field for all observation angles with transverse projection"""
    for obs_idx in range(obs_plane1):
        point = surface1_coordinates_ti[obs_idx]
        length = tm.length(point)
        length_xy = tm.length(vec2(point[0],point[1]))
        theta = ti.acos(point[0]/length_xy)
        phi_val = ti.acos(point[2]/length)
        
        sin_th, cos_th = ti.sin(theta), ti.cos(theta)
        sin_ph, cos_ph = ti.sin(phi_val), ti.cos(phi_val)
        
        rhat = vec3(sin_th * cos_ph, sin_th * sin_ph, cos_th)
        
        theta_hat = vec3(cos_th * cos_ph, cos_th * sin_ph, -sin_th)
        phi_hat   = vec3(-sin_ph, cos_ph, 0.0)
        
        E_total = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        for m in range(N):
            m_int = ti.cast(m, ti.i32)
            E_m = far_field(rhat, m_int, length)
            for j in ti.static(range(3)):
                E_total[j, 0] += E_m[j, 0]
                E_total[j, 1] += E_m[j, 1]
        
        E_th_comp = vec2(0.0, 0.0)
        E_ph_comp = vec2(0.0, 0.0)
        
        for j in ti.static(range(3)):
            E_th_comp += vec2(E_total[j, 0] * theta_hat[j], E_total[j, 1] * theta_hat[j])
            E_ph_comp += vec2(E_total[j, 0] * phi_hat[j], E_total[j, 1] * phi_hat[j])

        for j in ti.static(range(3)):
            E_field_s1[obs_idx, j] = vec2(E_total[j, 0], E_total[j, 1])
        

        mag_sq = (E_th_comp[0]**2 + E_th_comp[1]**2) + (E_ph_comp[0]**2 + E_ph_comp[1]**2)
        E_magnitude_s1[obs_idx] = ti.sqrt(mag_sq)

compute_f1()

@ti.kernel
def compute_f2():
    """Compute far field for all observation angles with transverse projection"""
    for obs_idx in range(obs_plane2):
        point = surface2_coordinates_ti[obs_idx]
        length = tm.length(point)
        length_xy = tm.length(vec2(point[0],point[1]))
        theta = ti.acos(point[0]/length_xy)
        phi_val = ti.acos(point[2]/length)
        
        sin_th, cos_th = ti.sin(theta), ti.cos(theta)
        sin_ph, cos_ph = ti.sin(phi_val), ti.cos(phi_val)
        
        rhat = vec3(sin_th * cos_ph, sin_th * sin_ph, cos_th)
        
        theta_hat = vec3(cos_th * cos_ph, cos_th * sin_ph, -sin_th)
        phi_hat   = vec3(-sin_ph, cos_ph, 0.0)
        
        E_total = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        for m in range(N):
            m_int = ti.cast(m, ti.i32)
            E_m = far_field(rhat, m_int, length)
            for j in ti.static(range(3)):
                E_total[j, 0] += E_m[j, 0]
                E_total[j, 1] += E_m[j, 1]
        
        E_th_comp = vec2(0.0, 0.0)
        E_ph_comp = vec2(0.0, 0.0)
        
        for j in ti.static(range(3)):
            E_th_comp += vec2(E_total[j, 0] * theta_hat[j], E_total[j, 1] * theta_hat[j])
            E_ph_comp += vec2(E_total[j, 0] * phi_hat[j], E_total[j, 1] * phi_hat[j])

        for j in ti.static(range(3)):
            E_field_s2[obs_idx, j] = vec2(E_total[j, 0], E_total[j, 1])
        

        mag_sq = (E_th_comp[0]**2 + E_th_comp[1]**2) + (E_ph_comp[0]**2 + E_ph_comp[1]**2)
        E_magnitude_s2[obs_idx] = ti.sqrt(mag_sq)

compute_f2()

@ti.kernel
def get_f1():
    for i in range(N1_coordinates):
        f1[i] = E_field_s1[i, 0]
        f1[i+N1_coordinates] = E_field_s1[i, 1]

get_f1()        

@ti.kernel
def get_f2():
    for i in range(N2_coordinates):
        f2[i] = E_field_s2[i, 0]
        f2[i+N2_coordinates] = E_field_s2[i, 1]

get_f2()


Axy1 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N1_coordinates, N_quad))
Ayx1 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N1_coordinates, N_quad))

@ti.kernel
def get_Axy1():
    for m in range(0,N1_coordinates):
        for n in range(0,N_quad):
            p0 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][0]]
            p1 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][1]]
            p2 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][2]]
            p3 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][3]]
            obv_point = surface1_coordinates_ti[m]

            area1 = 0.5 * tm.length(tm.cross(p1 - p0, p2 - p0))
            area2 = 0.5 * tm.length(tm.cross(p3 - p0, p2 - p0))

            result1 = vec2(0.0,0.0)
            result2 = vec2(0.0,0.0)
            for i in range(0,nop):
                location1 = p0*alpha[i] + p1*beta[i] + p2*gamma[i]
                result1 += green_func_derivative(obv_point, location1, 2)*w_gauss[i]

                location2 = p0*alpha[i] + p2*beta[i] + p3*gamma[i]
                result2 += green_func_derivative(obv_point, location2, 2)*w_gauss[i]

            result = 2*(result2*area2 + result1*area1)
            Axy1[m,n] = result
            Ayx1[m,n] = -1*result

get_Axy1()

Axy2 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N1_coordinates, N_quad))
Ayx2 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N1_coordinates, N_quad))

@ti.kernel
def get_Axy2():
    for m in range(0,N2_coordinates):
        for n in range(0,N_quad):
            p0 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][0]]
            p1 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][1]]
            p2 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][2]]
            p3 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][3]]
            obv_point = surface2_coordinates_ti[m]

            area1 = 0.5 * tm.length(tm.cross(p1 - p0, p2 - p0))
            area2 = 0.5 * tm.length(tm.cross(p3 - p0, p2 - p0))

            result1 = vec2(0.0,0.0)
            result2 = vec2(0.0,0.0)
            for i in range(0,nop):
                location1 = p0*alpha[i] + p1*beta[i] + p2*gamma[i]
                result1 += green_func_derivative(obv_point, location1, 2)*w_gauss[i]

                location2 = p0*alpha[i] + p2*beta[i] + p3*gamma[i]
                result2 += green_func_derivative(obv_point, location2, 2)*w_gauss[i]

            result = 2*(result2*area2 + result1*area1)
            Axy2[m,n] = result
            Ayx2[m,n] = -1*result

get_Axy2()


Axy1_np = Axy1.to_numpy()
Axy1_complex = Axy1_np[:, :, 0] + 1j * Axy1_np[:, :, 1]
Axy1_complex = np.round(Axy1_complex, 12)

Ayx1_np = Ayx1.to_numpy()
Ayx1_complex = Ayx1_np[:, :, 0] + 1j * Ayx1_np[:, :, 1]
Ayx1_complex = np.round(Ayx1_complex, 12)

Axy2_np = Axy2.to_numpy()
Axy2_complex = Axy2_np[:, :, 0] + 1j * Axy2_np[:, :, 1]
Axy2_complex = np.round(Axy2_complex, 12)

Ayx2_np = Ayx2.to_numpy()
Ayx2_complex = Ayx2_np[:, :, 0] + 1j * Ayx2_np[:, :, 1]
Ayx2_complex = np.round(Ayx2_complex, 12)

Ns, Nd = Axy1_complex.shape   

zero_block = np.zeros((Ns, Nd), dtype=Axy1_complex.dtype)

A1 = np.block([
    [zero_block,  Axy1_complex],
    [Ayx1_complex,         zero_block]
])

Ns, Nd = Axy2_complex.shape   

zero_block = np.zeros((Ns, Nd), dtype=Axy2_complex.dtype)

A2 = np.block([
    [zero_block,  Axy2_complex],
    [Ayx2_complex,         zero_block]
])

f1_np = f1.to_numpy()
f1_complex = f1_np[:, 0] + 1j * f1_np[:, 1]
f1_complex = np.round(f1_complex, 12)

f2_np = f2.to_numpy()
f2_complex = f2_np[:, 0] + 1j * f2_np[:, 1]
f2_complex = np.round(f2_complex, 12)

print("Done getting A1, A2, f1, f2.")
print(A1.shape)
print(f1_complex.shape)
A1_H = A1.conj().T
A2_H = A2.conj().T
# un = np.matmul(A1_H,f1_complex)

f1_abs = np.abs(f1_complex)
f2_abs = np.abs(f2_complex)

eta1 = (np.linalg.norm(f1_abs**2))**2
eta2 = (np.linalg.norm(f2_abs**2))**2

u_BP = A1_H @ f1_abs
zetas = np.linspace(0.01, 10, 1000)
costs = []
for zeta in zetas:
    u_test = zeta * u_BP
    r1_test = np.abs(A1 @ u_test)**2 - f1_abs**2
    r2_test = np.abs(A2 @ u_test)**2 - f2_abs**2
    costs.append(eta1 * np.linalg.norm(r1_test)**2 + eta2 * np.linalg.norm(r2_test)**2)

zeta = zetas[np.argmin(costs)]
un = zeta * u_BP

tolerance_factor = 1e-10
error = 1e3
area_d = del_y*n_cols*del_x*n_rows

eta1_A1H_2 = 2*eta1*A1_H
eta2_A2H_2 = 2*eta2*A2_H

print("eta1A1 shape: ",eta1_A1H_2.shape)
print("eta2A2 shape: ", eta2_A2H_2.shape)

gn_previous = 0
gn = 0
vn = 0

alpha_n = 1
Nd = N_quad 
beta_n = 0
count = 0

def line_search(un, vn, A1, A2, f1_abs, f2_abs, eta1, eta2):
    alpha = 1.0
    rho = 0.5      
    max_iter = 50
    
    r1 = np.abs(A1 @ un)**2 - f1_abs**2
    r2 = np.abs(A2 @ un)**2 - f2_abs**2
    c0 = eta1 * np.linalg.norm(r1)**2 + eta2 * np.linalg.norm(r2)**2
    
    for _ in range(max_iter):
        un_new = un + alpha * vn   
        r1_new = np.abs(A1 @ un_new)**2 - f1_abs**2
        r2_new = np.abs(A2 @ un_new)**2 - f2_abs**2
        c_new = eta1 * np.linalg.norm(r1_new)**2 + eta2 * np.linalg.norm(r2_new)**2
        
        if c_new < c0:
            return alpha
        alpha *= rho
    
    return alpha

while error > tolerance_factor:
    r1 = (np.abs(np.matmul(A1,un)))**2 - f1_abs**2
    r2 = (np.abs(np.matmul(A2,un)))**2 - f2_abs**2
    c1 = eta1*((np.linalg.norm(r1))**2)
    c2 = eta2*((np.linalg.norm(r2))**2)

    Mx = un[:Nd]          
    My = un[Nd:]         

    Mx_2d = Mx.reshape(n_rows, n_cols)  
    My_2d = My.reshape(n_rows, n_cols) 

    grad_y_x, grad_x_x = np.gradient(Mx_2d, del_y, del_x)
    grad_mag_sq_x = np.abs(grad_x_x)**2 + np.abs(grad_y_x)**2

    grad_y_y, grad_x_y = np.gradient(My_2d, del_y, del_x)
    grad_mag_sq_y = np.abs(grad_x_y)**2 + np.abs(grad_y_y)**2

    grad_mag_sq = grad_mag_sq_x + grad_mag_sq_y

    del_sqr = (c1 + c2) / (2 * del_x * del_y)
    b_n = (1 / np.sqrt(area_d)) * (grad_mag_sq + del_sqr)**(-0.5) 

    b_n_sq = b_n**2

    div_x_mx = np.gradient(b_n_sq * grad_x_x, del_x, axis=1)
    div_y_mx = np.gradient(b_n_sq * grad_y_x, del_y, axis=0)
    g_MR_x = -(div_x_mx + div_y_mx)  

    div_x_my = np.gradient(b_n_sq * grad_x_y, del_x, axis=1)
    div_y_my = np.gradient(b_n_sq * grad_y_y, del_y, axis=0)
    g_MR_y = -(div_x_my + div_y_my)
    g_MR = np.concatenate([g_MR_x.flatten(), g_MR_y.flatten()])

    term1 = eta1_A1H_2 @ (r1 * (A1 @ un))  
    term2 = eta2_A2H_2 @ (r2 * (A2 @ un))  
    term3 = (c1 + c2)*g_MR

    gn_previous = gn
    gn = term1 + term2 + term3

    g_flat = gn.conj().flatten()
    diff_flat = (gn - gn_previous).flatten()
    numerator = np.dot(g_flat, diff_flat)          
    denominator = np.linalg.norm(gn_previous)**2       
    if count != 0:
        beta_n = numerator / denominator
    else: 
        beta_n = 0

    un_previous = un
    vn = -gn + beta_n * vn
    
    alpha_n = line_search(un, vn, A1, A2, f1_abs, f2_abs, eta1, eta2)
    # alpha_n = 1
    # un = un + alpha_n * vn 
    error = np.abs(np.linalg.norm(un-un_previous)/np.linalg.norm(un))
    count += 1

    if count > 10000:
        print("Not Converging",error)
        break

print(error, count)

Mx_2d = un[:N_quad].reshape(n_rows, n_cols)
My_2d = un[N_quad:].reshape(n_rows, n_cols)

plot_data = np.abs(Mx_2d)**2 + np.abs(My_2d)**2
plot_data = plot_data/np.max(plot_data)
plot_data = 20*np.log10(plot_data)
cell_data = plot_data.ravel(order='C')

n_cells = reconstructed_quad.shape[0]
cells = np.hstack([np.full((n_cells, 1), 4), reconstructed_quad]).astype(np.int64).flatten()
celltypes = np.full(n_cells, pv.CellType.QUAD)

pts = reconstructed_coordinates.copy()
pts[:, 2] = 0.0   

grid = pv.UnstructuredGrid(cells, celltypes, pts)
grid.cell_data["|M|²"] = cell_data

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="|M|²", cmap="jet", show_edges=False, show_scalar_bar=False)
plotter.add_scalar_bar(title="|M|²", n_labels=5, fmt="%.2e")
plotter.view_xy()  
plotter.show()
plotter.close()
del plotter