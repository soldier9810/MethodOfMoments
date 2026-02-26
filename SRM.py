import taichi as ti
import taichi.math as tm
import numpy as np
import meshio
from collections import defaultdict
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu, default_fp=ti.f64)
vec2 = ti.types.vector(2, ti.f64)
vec3 = ti.types.vector(3, ti.f64)

nop = 3
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

wavenumber = 1

############################################################# READING AND ORGANIZING MESH

mesh = meshio.read(r"meshes/comparison_quad.msh")

reconstructed_coordinates = mesh.points.astype(np.float64)
reconstructed_quad = mesh.cells_dict["quad"].astype(np.int32)
for i in range(len(reconstructed_quad)):
    quad = reconstructed_quad[i]
    pts = reconstructed_coordinates[quad]  # shape (4, 3)
    
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)  # counter-clockwise order
    
    reconstructed_quad[i] = quad[sorted_indices]

centroids = reconstructed_coordinates[reconstructed_quad[:,0]]#.mean(axis=1)  # (N, 3)
cx = np.round(centroids[:, 0], decimals=6)
cy = np.round(centroids[:, 1], decimals=6)

sorted_indices = np.lexsort((cx, cy))
reconstructed_quad = reconstructed_quad[sorted_indices]

unique_y = np.unique(np.round(cy, decimals=6))
unique_x = np.unique(np.round(cx, decimals=6))

n_rows = len(unique_y)
n_cols = len(unique_x)

print(n_rows, n_cols)

reconstructed_quad_2d = reconstructed_quad.reshape(n_rows, n_cols, 4)

#########################################################################

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

surface1_coordinates = reconstructed_coordinates + [0,0,1]
surface2_coordinates = reconstructed_coordinates + [0,0,2]

surface1_quad = mesh.cells_dict["quad"].astype(np.int32)
surface2_quad = mesh.cells_dict["quad"].astype(np.int32)

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

N_quad = len(reconstructed_quad)
N1_quad = len(surface1_quad)
N2_quad = len(surface2_quad)

N_coordinates = len(reconstructed_coordinates)
N1_coordinates = len(surface1_coordinates)
N2_coordinates = len(surface2_coordinates)

@ti.func
def c_exp_j(theta: float) -> vec2:
    return vec2(ti.cos(theta), ti.sin(theta))


complex_electric_field = ti.types.struct(x = vec2,
                                         y = vec2,
                                         z = vec2)

@ti.func
def electric_field(r: vec3) -> complex_electric_field:
    phase = c_exp_j(-wavenumber * r[2])
    result = complex_electric_field(
        x = vec2(1.0 * phase[0], 1.0 * phase[1]),
        y = vec2(0,0),
        z = vec2(0,0)
    )
    return result

# r is observation point and rprime is source 
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

@ti.kernel
def get_f1():
    for i in range(N1_coordinates):
        calc = electric_field(surface1_coordinates_ti[i])
        f1[i] = calc.x
        f1[i+N1_coordinates] = calc.y

get_f1()        

@ti.kernel
def get_f2():
    for i in range(N2_coordinates):
        calc = electric_field(surface2_coordinates_ti[i])
        f2[i] = calc.x
        f2[i+N2_coordinates] = calc.y

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

Ns, Nd = Axy1_complex.shape   # Ns = measurement points, Nd = reconstruction cells

zero_block = np.zeros((Ns, Nd), dtype=Axy1_complex.dtype)

A1 = np.block([
    [zero_block,  Axy1_complex],
    [Ayx1_complex,         zero_block]
])

Ns, Nd = Axy2_complex.shape   # Ns = measurement points, Nd = reconstruction cells

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

u_BP = A1_H @ f1_abs          # back propagation
# Find zeta that minimizes C(zeta * u_BP) - simple line search
zetas = np.linspace(0.01, 10, 1000)
costs = []
for zeta in zetas:
    u_test = zeta * u_BP
    r1_test = np.abs(A1 @ u_test)**2 - f1_abs**2
    r2_test = np.abs(A2 @ u_test)**2 - f2_abs**2
    costs.append(eta1 * np.linalg.norm(r1_test)**2 + eta2 * np.linalg.norm(r2_test)**2)

zeta = zetas[np.argmin(costs)]
un = zeta * u_BP

tolerance_factor = 1e-3
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

Nd = N_quad  # 100

Mx = un[:Nd]          # first half = Mx coefficients
My = un[Nd:]          # second half = My coefficients

Mx_2d = Mx.reshape(n_rows, n_cols)  # (10, 10)
My_2d = My.reshape(n_rows, n_cols)  # (10, 10)
beta_n = 0
count = 0
while error > tolerance_factor:
    r1 = (np.abs(np.matmul(A1,un)))**2 - f1_abs**2
    r2 = (np.abs(np.matmul(A2,un)))**2 - f2_abs**2
    c1 = eta1*((np.linalg.norm(r1))**2)
    c2 = eta2*((np.linalg.norm(r2))**2)

    # For Mx
    grad_y_x, grad_x_x = np.gradient(Mx_2d, del_y, del_x)
    grad_mag_sq_x = np.abs(grad_x_x)**2 + np.abs(grad_y_x)**2

    # For My
    grad_y_y, grad_x_y = np.gradient(My_2d, del_y, del_x)
    grad_mag_sq_y = np.abs(grad_x_y)**2 + np.abs(grad_y_y)**2

    # Combined gradient magnitude for b_n (over both components)
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

    # term1 = np.matmul(eta1_A1H_2, r1) * np.matmul(A1, un)
    # term2 = np.matmul(eta2_A2H_2, r2) * np.matmul(A2, un)
    term1 = eta1_A1H_2 @ (r1 * (A1 @ un))  # hadamard inside, then matrix multiply
    term2 = eta2_A2H_2 @ (r2 * (A2 @ un))  # hadamard inside, then matrix multiply
    term3 = (c1 + c2)*g_MR

    gn_previous = gn
    gn = term1 + term2 + term3

    g_flat = gn.conj().flatten()
    diff_flat = (gn - gn_previous).flatten()
    numerator = np.dot(g_flat, diff_flat)           # g_n^H · (g_n - g_{n-1})
    denominator = np.linalg.norm(gn_previous)**2        # ||g_{n-1}||_D²
    if count != 0:
        beta_n = numerator / denominator
    else: beta_n = 0

    vn = vn*beta_n + gn
    un_previous = un
    un = un + alpha_n*vn
    error = np.abs(np.linalg.norm(un-un_previous)/np.linalg.norm(un))
    count += 1

    if count > 1000:
        print(error)
        break