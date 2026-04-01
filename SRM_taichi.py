import taichi as ti
import numpy as np
import pandas as pd
import taichi.math as tm
import meshio
import gc
from datetime import datetime
from datetime import timedelta
import time
import math

ti.init(arch = ti.gpu, default_fp = ti.f64)
vec2 = ti.types.vector(2, ti.f64)
vec3 = ti.types.vector(3, ti.f64)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.perf_counter()

nop = 7
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


wavelength = 13
wavenumber = 2*np.pi/wavelength

############################################################# READING AND ORGANIZING MESH

# mesh = meshio.read(r"meshes/comparison_quad.msh")
mesh = meshio.read(r"horn_parab_1_28_larger.msh")
reconstructed_coordinates = mesh.points.astype(np.float64)
reconstructed_quad = mesh.cells_dict["quad"].astype(np.int32)

del mesh
gc.collect()

for i in range(len(reconstructed_quad)):
    quad = reconstructed_quad[i]
    pts = reconstructed_coordinates[quad]  # shape (4, 3)

    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)  # counter-clockwise order

    reconstructed_quad[i] = quad[sorted_indices]

centroids = reconstructed_coordinates[reconstructed_quad[:,0]]
cx = np.round(centroids[:, 0], decimals=6)
cy = np.round(centroids[:, 1], decimals=6)

sorted_indices = np.lexsort((cx, cy))
reconstructed_quad = reconstructed_quad[sorted_indices]

min_corner = np.min(reconstructed_coordinates, axis=0)
max_corner = np.max(reconstructed_coordinates, axis=0)
print(min_corner, max_corner)
print(len(reconstructed_quad))
mesh_del_x = abs(reconstructed_coordinates[reconstructed_quad[0][1]][0] - reconstructed_coordinates[reconstructed_quad[0][0]][0])
mesh_del_y = abs(reconstructed_coordinates[reconstructed_quad[0][2]][1] - reconstructed_coordinates[reconstructed_quad[0][1]][1])
print(mesh_del_x, mesh_del_y)
n_rows = math.ceil((max_corner[1] - min_corner[1])/mesh_del_y)
n_cols = math.floor((max_corner[0] - min_corner[0])/mesh_del_x)

print(n_rows, n_cols)
N_quad = len(reconstructed_quad)
N_coordinates = len(reconstructed_coordinates)
reconstructed_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(reconstructed_coordinates)))
reconstructed_quad_ti = ti.Vector.field(n=4, dtype=ti.int32, shape=(len(reconstructed_quad)))

for i in range(0,len(reconstructed_coordinates)):
    reconstructed_coordinates_ti[i] = reconstructed_coordinates[i]

for i in range(0,len(reconstructed_quad)):
    reconstructed_quad_ti[i] = reconstructed_quad[i]


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



surface1_measurements = pd.read_csv(
    r"Elec_field/latest/horn_arn_220.txt",
    sep=r"\s+",      # split on whitespace
    skiprows=[0,1],       # skip the dashed line
    engine="python",
    names=["x_mm", "y_mm", "z_mm",
           "ExRe", "ExIm",
           "EyRe", "EyIm",
           "EzRe", "EzIm"]
)

surface1_coordinates = surface1_measurements[["x_mm", "y_mm", "z_mm"]].to_numpy()
surface1_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(surface1_coordinates)))
N1_coordinates = len(surface1_coordinates)
#for i in range(0,len(surface1_coordinates)):
 #   surface1_coordinates_ti[i] = surface1_coordinates[i]

surface1_coordinates_ti.from_numpy(surface1_coordinates)

Ex = surface1_measurements["ExRe"].to_numpy() + 1j * surface1_measurements["ExIm"].to_numpy()
Ey = surface1_measurements["EyRe"].to_numpy() + 1j * surface1_measurements["EyIm"].to_numpy()
Ex = np.abs(Ex)
Ey = np.abs(Ey)
f1 = np.concatenate([Ex, Ey])
#f1 = f1 / np.linalg.norm(f1)
eta1 = (np.linalg.norm(f1**2))**2
f1_ti = ti.field(dtype=ti.f64, shape=(2*N1_coordinates,))
f1_ti.from_numpy(f1)


del surface1_measurements, Ex, Ey, surface1_coordinates, f1
gc.collect()



surface2_measurements = pd.read_csv(
    r"Elec_field/latest/horn_arn_250.txt",
    sep=r"\s+",      # split on whitespace
    skiprows=[0,1],       # skip the dashed line
    engine="python",
    names=["x_mm", "y_mm", "z_mm",
           "ExRe", "ExIm",
           "EyRe", "EyIm",
           "EzRe", "EzIm"]
)
surface2_coordinates = surface2_measurements[["x_mm", "y_mm", "z_mm"]].to_numpy()
surface2_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(surface2_coordinates)))
#for i in range(0,len(surface2_coordinates)):
 #   surface2_coordinates_ti[i] = surface2_coordinates[i]
surface2_coordinates_ti.from_numpy(surface2_coordinates)

N2_coordinates = len(surface2_coordinates)
Ex2 = surface2_measurements["ExRe"].to_numpy() + 1j * surface2_measurements["ExIm"].to_numpy()
Ey2 = surface2_measurements["EyRe"].to_numpy() + 1j * surface2_measurements["EyIm"].to_numpy()
Ex2 = np.abs(Ex2)
Ey2 = np.abs(Ey2)
f2 = np.concatenate([Ex2, Ey2])
#f2 = f2/np.linalg.norm(f2)
eta2 = (np.linalg.norm(f2**2))**2
f2_ti = ti.field(dtype=ti.f64, shape=(2*N2_coordinates,))
f2_ti.from_numpy(f2)
del surface2_measurements, Ex2, Ey2, surface2_coordinates, f2
gc.collect()

print(f2_ti.shape[0])

Axy1 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N1_coordinates, N_quad))
# Ayx1 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N1_coordinates, N_quad))

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
            # Ayx1[m,n] = -1*result

get_Axy1()

Axy2 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N2_coordinates, N_quad))
# Ayx2 = ti.Vector.field(n=2, dtype=ti.f64, shape=(N2_coordinates, N_quad))

@ti.kernel
def get_Axy2():
    for m in range(0, N2_coordinates):
        for n in range(0, N_quad):
            p0 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][0]]
            p1 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][1]]
            p2 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][2]]
            p3 = reconstructed_coordinates_ti[reconstructed_quad_ti[n][3]]
            obv_point = surface2_coordinates_ti[m]

            area1 = 0.5 * tm.length(tm.cross(p1 - p0, p2 - p0))
            area2 = 0.5 * tm.length(tm.cross(p3 - p0, p2 - p0))

            result1 = vec2(0.0, 0.0)
            result2 = vec2(0.0, 0.0)
            for i in range(0, nop):
                location1 = p0*alpha[i] + p1*beta[i] + p2*gamma[i]
                result1 += green_func_derivative(obv_point, location1, 2)*w_gauss[i]

                location2 = p0*alpha[i] + p2*beta[i] + p3*gamma[i]
                result2 += green_func_derivative(obv_point, location2, 2)*w_gauss[i]

            result = 2*(result2*area2 + result1*area1)
            Axy2[m, n] = result
            # Ayx2[m,n] = -1*result

get_Axy2()

un = ti.Vector.field(n=2, dtype=ti.f64, shape=(2*N_quad,))
print(un.shape)

A_un_mul_abs = ti.field(dtype=ti.f64, shape=(2*N1_coordinates,))

@ti.kernel
def matmul(A: ti.template(), b: ti.template(), c: ti.template()):
    b_shape0 = ti.cast(b.shape[0]/2, ti.i32)
    for i in range(0, A.shape[0]):
        c[i] = vec2(0, 0)
        c[i+A.shape[0]] = vec2(0, 0)
        for j in range(0, b_shape0):
            c[i] += tm.cmul(A[i, j], b[b_shape0 + j])
            c[i+A.shape[0]] += tm.cmul(-A[i, j], b[j])
'''
@ti.kernel
def matmul_abs(A: ti.template()):
    for i in range(0, N1_coordinates):
        A_un_mul_abs[i] = 0
        A_un_mul_abs[i + N1_coordinates] = 0
        for j in range(0, N_quad):
            A_un_mul_abs[i] += tm.cmul(A[i, j], un[N_quad + j]).norm_sqr()
            A_un_mul_abs[i + N1_coordinates] += tm.cmul(-A[i, j], un[j]).norm_sqr()
'''
@ti.kernel
def matmul_abs(A: ti.template()):
    for i in range(0, N1_coordinates):
        # accumulate complex sum first
        row_sum_ex = vec2(0.0, 0.0)
        row_sum_ey = vec2(0.0, 0.0)
        for j in range(0, N_quad):
            row_sum_ex += tm.cmul(A[i, j], un[N_quad + j])   # Ex = Axy * My
            row_sum_ey += tm.cmul(-A[i, j], un[j])           # Ey = -Axy * Mx
        # then take squared magnitude
        A_un_mul_abs[i]               = row_sum_ex.norm_sqr()
        A_un_mul_abs[i + N1_coordinates] = row_sum_ey.norm_sqr()
@ti.kernel
def matmul_hermitian(A: ti.template(), b: ti.template(), c: ti.template()):
    b_shape0 = ti.cast(b.shape[0]/2, ti.i32)
    for i in range(0, A.shape[1]):
        c[i] = vec2(0, 0)
        c[i+A.shape[1]] = vec2(0, 0)
        for j in range(0, b_shape0):
            c[i] += vec2(A[i, j].x, -A[i, j].y)*b[b_shape0 + j]
            c[i+A.shape[1]] += vec2(-A[i, j].x, A[i, j].y)*b[j]


matmul_hermitian(Axy1, f1_ti, un)
matmul_abs(Axy1)

sum_c_Aun = ti.field(dtype=ti.f64, shape=())
sum_c_fA = ti.field(dtype=ti.f64, shape=())
zeta_coeff = ti.field(dtype=ti.f64, shape=())
const_coeff = ti.field(dtype=ti.f64, shape=())
'''
@ti.kernel
def sum_field(f: ti.template()):
    sum_c_Aun[None] = 0
    sum_c_fA[None] = 0
    for i in range(0, 2*N1_coordinates):
        sum_c_Aun[None] += A_un_mul_abs[i]
        sum_c_fA[None] += A_un_mul_abs[i]*f[i]
'''
@ti.kernel
def sum_field(f: ti.template()):
    sum_c_Aun[None] = 0.0
    sum_c_fA[None] = 0.0
    for i in range(0, 2*N1_coordinates):
        Aun_sq = A_un_mul_abs[i]              # |A·uBP|²[i]
        fi_sq  = f[i] * f[i]                  # |f|²[i]
        sum_c_Aun[None] += Aun_sq * Aun_sq    # ‖|A·uBP|²‖²
        sum_c_fA[None]  += Aun_sq * fi_sq     # ⟨|A·uBP|², |f|²⟩
sum_field(f1_ti)
print(sum_c_Aun)
zeta_coeff[None] = sum_c_Aun[None]*eta1
const_coeff[None] = sum_c_fA[None]*eta1
matmul_abs(Axy2)
sum_field(f2_ti)
print(sum_c_Aun)
zeta_coeff[None] += sum_c_Aun[None]*eta2
const_coeff[None] += sum_c_fA[None]*eta2
print(zeta_coeff)
zeta_val = math.sqrt(float(const_coeff[None])/float(zeta_coeff[None]))
print(zeta_val)
zeta_val_ti = ti.field(dtype=ti.f64, shape=())

@ti.kernel
def get_u0():
    zeta_val_ti[None] = ti.cast(zeta_val, ti.f64)
    for i in range(0, 2*N_quad):
        un[i] = zeta_val_ti[None]*un[i]

duration = timedelta(seconds=time.perf_counter()-start_time)
print(duration)
