import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import meshio
from collections import defaultdict
import pyvista as pv
import time
from datetime import timedelta
from tqdm import tqdm

import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, default_fp=ti.f64)

vec2 = ti.types.vector(2, ti.f64)
vec3 = ti.types.vector(3, ti.f64)
vec3_int = ti.types.vector(3, ti.i32)

start_time = time.perf_counter()
c = 3e8
# f = 112e8
# wavelength = c/f
wavelength = 1
f = c/wavelength
k = 2*np.pi/wavelength
R = np.linspace(0.01, 2, 500)
H0 = special.hankel1(0, k * R)
w = 2*np.pi*f
u = 4*np.pi*1e-7
epsilon = 8.85*1e-12

mesh = meshio.read(r"plane50points.msh")
points_np = mesh.points.astype(np.float64)
triangles_np = mesh.cells_dict["triangle"].astype(np.int32)

print("Triangles:", len(triangles_np))

for i in range(0,len(points_np)):
    points_np[i] = (np.array(points_np[i])/1e6)
for i in range(0,len(triangles_np)):
    triangles_np[i] = np.array(triangles_np[i])

min_corner = np.min(points_np, axis=0)
max_corner = np.max(points_np, axis=0)
bbox_center = (min_corner + max_corner) / 2.0
points_np -= bbox_center

avg_side_length = 0
for tri in points_np[triangles_np]:
    s1 = np.linalg.norm(tri[0] - tri[1])
    s2 = np.linalg.norm(tri[0] - tri[2])
    s3 = np.linalg.norm(tri[2] - tri[1])
    avg_side_length += (s1 + s2 + s3) / 3

avg_side_length = avg_side_length / len(triangles_np)

# Create Taichi fields
n_points = points_np.shape[0]
points_ti = ti.Vector.field(3, dtype=ti.f64, shape=n_points)

n_tri = triangles_np.shape[0]
triangles_ti = ti.Vector.field(3, dtype=ti.i32, shape=n_tri)

for i in range(n_points):
    points_ti[i] = points_np[i]  

for i in range(n_tri):
    triangles_ti[i] = triangles_np[i]   

nop = 16
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
############# 16 point
elif nop == 16:
    alpha = ti.Vector([0.081414823414554, 0.081414823414554, 0.837170353641812,
                    0.658861384496480, 0.658861384496480, 0.316277231295461,
                    0.316277231295461, 0.024180199784358, 0.024180199784358,
                    0.024180199784358, 0.310352451033785, 0.636502499121399,
                    0.636502499121399, 0.636502499121399, 0.053145049844816,
                    0.053145049844816])

    beta  = ti.Vector([0.081414823414554, 0.837170353641812, 0.081414823414554,
                    0.316277231295461, 0.024180199784358, 0.658861384496480,
                    0.024180199784358, 0.658861384496480, 0.024180199784358,
                    0.316277231295461, 0.053145049844816, 0.053145049844816,
                    0.310352451033785, 0.310352451033785, 0.636502499121399,
                    0.636502499121399])

    gamma = 1.0 - alpha - beta

    w_gauss     = ti.Vector([0.090817990382754, 0.090817990382754, 0.090817990382754,
                    0.082851075618374, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.025731066440455, 0.025731066440455,
                    0.025731066440455, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.082851075618374, 0.025731066440455,
                    0.025731066440455])

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

@ti.func
def c_mul(a: vec2, b: vec2) -> vec2:
    return vec2(
        a[0] * b[0] - a[1] * b[1],
        a[0] * b[1] + a[1] * b[0]
    )
    
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
    ################ in, out
    sign_rp = 1 - 2 * rp_type
    sign_rq = 1 - 2 * rq_type
    rho_rp = sign_rp * (rp - rp_free)
    rho_rq = sign_rq * (rq - rq_free)
    final_result = vec2(0,0)
    if inner:
        distance = tm.length(rp - rq)
        first_term = vec2(0, w*u/4) * tm.dot(rho_rp, rho_rq) - vec2(0, 1/(w*epsilon)) * sign_rp * sign_rq
        eps = 1e-12
        if distance < eps:
            first_term *= vec2(0, -k)
        else:
            first_term *= (c_exp_j(-1 * k * distance) - vec2(1, 0)) / distance
        
        final_result = first_term
    else:
        tri0 = points_ti[triangles_ti[tin][0]]
        tri1 = points_ti[triangles_ti[tin][1]]
        tri2 = points_ti[triangles_ti[tin][2]]
        side11 = tri0 - tri1
        side12 = tri0 - tri2
        ncap = tm.cross(side11, side12)
        ncap /= tm.length(ncap)
        
        I1, I2 = get_I1I2(tin, rp, ncap)
        
        third_term = I2 * vec2(0, 1 / (w * epsilon * sign_rp * sign_rq))
        second_term = vec2(tm.dot(rho_rp, sign_rq * (I1 + (rp - rq_free) * I2)), 0)
        
        final_result = (second_term - third_term) / triangle_properties[tin].area
    return final_result

@ti.func
def get_component_in_plane(ncap: vec3, r:vec3) -> vec3:
    return r - ncap * tm.dot(ncap, r)

@ti.func
def get_I1I2(tin: ti.i32, obv_point: vec3, ncap: vec3):
    tri0 = points_ti[triangles_ti[tin][0]]
    tri1 = points_ti[triangles_ti[tin][1]]
    tri2 = points_ti[triangles_ti[tin][2]]
    eps = 1e-12
    I1 = vec3(0.0,0.0,0.0)
    I2 = 0.0
    p1 = vec3(0.0,0.0,0.0)
    p2 = vec3(0.0,0.0,0.0)
    for i in range(0,3):
        if i == 2:
            p1 = tri0
            p2 = tri2
        elif i == 1:
            p1 = tri1
            p2 = tri2
        else:
            p1 = tri0
            p2 = tri1
        lcap = (p2 - p1)
        lcap = lcap/tm.length(lcap)
        ucap = tm.cross(lcap,ncap)      
        obv_point_p = obv_point + ucap*1e-12
        rho = get_component_in_plane(ncap, obv_point_p)
        rho_p1 = get_component_in_plane(ncap, p1)
        rho_p2 = get_component_in_plane(ncap, p2)
        l_plus = tm.dot((rho_p2 - rho),lcap)
        l_minus = tm.dot((rho_p1 - rho),lcap)
        p0 = ti.abs(tm.dot(ucap, rho_p1 - rho))
        p_plus = tm.length(rho_p2 - rho)
        p_minus = tm.length(rho_p1 - rho)
        p0_cap = ((rho_p1 - rho) - l_minus * lcap)
        p0_cap /= p0
        d = ti.abs(tm.dot(ncap, obv_point_p - p1))
        R0 = tm.sqrt(p0**2 + d**2)
        Rplus = tm.sqrt(p_plus**2 + d**2)
        Rminus = tm.sqrt(p_minus**2 + d**2)
        common_term = tm.log((Rplus + l_plus + eps) / (Rminus + l_minus + eps))
        value = ((R0**2)*common_term) + l_plus*Rplus - l_minus*Rminus
        I1 += value*ucap
        tan_term1 = tm.atan2(p0*l_plus,(R0**2 + d*Rplus + eps))
        tan_term2 = tm.atan2(p0*l_minus,(R0**2 + d*Rminus + eps))
        value = (p0*common_term) - d*(tan_term1 - tan_term2)
        value *= tm.dot(p0_cap,ucap)
        I2 += value

    I1 = I1/2
    return I1, I2

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
    ######## REMOVING AREA FROM THIS FUNCTION SINCE IT CANCELS OUT IN THE FINAL EQUATION
    final_result = vec2(0.0, 0.0)
    shared = get_triangle_relationship(tin, tout)
    if shared < 2:
        # Non-singular case
        for i in range(nop):  # Changed from range(0, len(w_gauss))
            location_out = alpha[i]*vout1 + beta[i]*vout2 + gamma[i]*vout3
            result = vec2(0.0, 0.0)
            for j in range(nop):  # Changed from range(0, len(w_gauss))
                location_in = alpha[j]*vin1 + beta[j]*vin2 + gamma[j]*vin3
                result += func(location_out, location_in, rp_type, rq_type, rp_free, rq_free) * w_gauss[j]
            final_result += result * w_gauss[i]
    else:
        # Singular case
        for i in range(nop):  # Changed from range(0, len(w_gauss))
            location_out = alpha[i]*vout1 + beta[i]*vout2 + gamma[i]*vout3
            result = vec2(0.0, 0.0)
            
            # Inner integration
            for j in range(nop):  # Changed from range(0, len(w_gauss))
                location_in = alpha[j]*vin1 + beta[j]*vin2 + gamma[j]*vin3
                result += func_singularity(location_out, location_in, rp_type, rq_type, 
                                          rp_free, rq_free, tout, tin, 1) * w_gauss[j]
            
            # Outer singularity term (doesn't depend on location_in)
            # Note: location_in here should probably be location_out or a specific point
            # I'm using location_out as it makes more sense
            result += func_singularity(location_out, location_out, rp_type, rq_type, 
                                      rp_free, rq_free, tout, tin, 0)
            
            final_result += result * w_gauss[i]
    
    return final_result

@ti.func
def electric_field(r: vec3) -> ti.types.matrix(3, 2, float):
    # Returns a 3x2 matrix where each row is a complex number (real, imag)
    phase = c_exp_j(-k * r[2])
    result = ti.Matrix([
        [1.0 * phase[0], 1.0 * phase[1]],  # x-component (complex)
        [0.0, 0.0],                         # y-component (complex)
        [0.0, 0.0]                          # z-component (complex)
    ])
    return result

@ti.func
def complex_dot(v_real: vec3, v_complex) -> vec2:
    # Dot product of real vec3 with complex vec3 (represented as 3x2 matrix)
    result = vec2(0.0, 0.0)
    for i in range(3):
        result[0] += v_real[i] * v_complex[i, 0]  # real part
        result[1] += v_real[i] * v_complex[i, 1]  # imaginary part
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
Z = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(N, N))  # Complex as vec2 (real, imag)
I = ti.field(dtype=ti.types.vector(2, ti.f64), shape=(N,))    # Complex as vec2

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
                0, 0, nth_basis.free_neg, mth_basis.free_neg, 
                mth_basis.neg_tri, nth_basis.neg_tri
            )
            
            Imneg_npos = double_integration(
                mth_basis.free_neg, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_pos, nth_basis.cp1, nth_basis.cp2, 
                0, 1, nth_basis.free_pos, mth_basis.free_neg, 
                mth_basis.neg_tri, nth_basis.pos_tri
            )
            
            Impos_nneg = double_integration(
                mth_basis.free_pos, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_neg, nth_basis.cp1, nth_basis.cp2, 
                1, 0, nth_basis.free_neg, mth_basis.free_pos, 
                mth_basis.pos_tri, nth_basis.neg_tri
            )
            
            Impos_npos = double_integration(
                mth_basis.free_pos, mth_basis.cp1, mth_basis.cp2, 
                nth_basis.free_pos, nth_basis.cp1, nth_basis.cp2, 
                1, 1, nth_basis.free_pos, mth_basis.free_pos, 
                mth_basis.pos_tri, nth_basis.pos_tri
            )
            
            factor = (mth_basis.edge_length * nth_basis.edge_length) / (4 * ti.math.pi)
            Z[m, n] = factor * (Imneg_nneg + Imneg_npos + Impos_nneg + Impos_npos)
        print(m)

get_Z()

# Convert Taichi fields to NumPy
Z_np = Z.to_numpy()
I_np = I.to_numpy()

# Convert to complex arrays
Z_complex = Z_np[:, :, 0] + 1j * Z_np[:, :, 1]
I_complex = I_np[:, 0] + 1j * I_np[:, 1]

# Round for numerical stability
Z_complex = np.round(Z_complex, 12)
I_complex = np.round(I_complex, 12)

# Solve the linear system
coeff = np.linalg.solve(Z_complex, I_complex)

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
duration = timedelta(seconds=time.perf_counter()-start_time)
print(duration)

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
vmin, vmax = 0, mag.max()  # or fixed global range
mesh.save("mesh50_taichi.vtk")