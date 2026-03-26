import taichi as ti
import taichi.math as tm
import numpy as np
import meshio
from collections import defaultdict
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import gc

ti.init(arch=ti.cpu, default_fp=ti.f64)
vec2 = ti.types.vector(2, ti.f64)
vec3 = ti.types.vector(3, ti.f64)

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
mesh = meshio.read(r"horn_parab_1_3.msh")
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

centroids = reconstructed_coordinates[reconstructed_quad[:,0]]#.mean(axis=1)  # (N, 3)
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
n_rows = int((max_corner[1] - min_corner[1])/mesh_del_y)
n_cols = int((max_corner[0] - min_corner[0])/mesh_del_x)

print(n_rows, n_cols)

#########################################################################

p0 = reconstructed_coordinates[reconstructed_quad[0][0]]
p1 = reconstructed_coordinates[reconstructed_quad[0][1]]
del_x = np.linalg.norm(p0-p1)
#del_x = round(del_x, 2)
p2 = reconstructed_coordinates[reconstructed_quad[0][2]]
del_y = np.linalg.norm(p2-p1)
#del_y = round(del_y, 2)
print(del_x, del_y)
reconstructed_coordinates_ti = ti.Vector.field(n=3, dtype=ti.f64, shape=(len(reconstructed_coordinates)))
reconstructed_quad_ti = ti.Vector.field(n=4, dtype=ti.int32, shape=(len(reconstructed_quad)))

for i in range(0,len(reconstructed_coordinates)):
    reconstructed_coordinates_ti[i] = reconstructed_coordinates[i]

for i in range(0,len(reconstructed_quad)):
    reconstructed_quad_ti[i] = reconstructed_quad[i]

###########################################################################
#r"Elec_field/HORNANTENNA_PARABOLIC REFLECTOR/HORN_120_LARGERPLANE.txt",
surface1_measurements = pd.read_csv(
    r"Elec_field/HORNANTENNA_PARABOLIC REFLECTOR/HORN_200_LARGERPLANE.txt",
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

for i in range(0,len(surface1_coordinates)):
    surface1_coordinates_ti[i] = surface1_coordinates[i]

Ex = surface1_measurements["ExRe"].to_numpy() + 1j * surface1_measurements["ExIm"].to_numpy()
Ey = surface1_measurements["EyRe"].to_numpy() + 1j * surface1_measurements["EyIm"].to_numpy()
f1 = np.concatenate([Ex, Ey])
N1_coordinates = len(surface1_coordinates)

del surface1_measurements, Ex, Ey, surface1_coordinates
gc.collect()



surface2_measurements = pd.read_csv(
    r"Elec_field/HORNANTENNA_PARABOLIC REFLECTOR/HORN_240_LARGERPLANE.txt",
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
for i in range(0,len(surface2_coordinates)):
    surface2_coordinates_ti[i] = surface2_coordinates[i]

N2_coordinates = len(surface2_coordinates)
Ex2 = surface2_measurements["ExRe"].to_numpy() + 1j * surface2_measurements["ExIm"].to_numpy()
Ey2 = surface2_measurements["EyRe"].to_numpy() + 1j * surface2_measurements["EyIm"].to_numpy()
f2 = np.concatenate([Ex2, Ey2])

del surface2_measurements, Ex2, Ey2, surface2_coordinates
gc.collect()

N_quad = len(reconstructed_quad)
N_coordinates = len(reconstructed_coordinates)


@ti.func
def c_exp_j(theta: float) -> vec2:
    return vec2(ti.cos(theta), ti.sin(theta))


complex_electric_field = ti.types.struct(x = vec2,
                                         y = vec2,
                                         z = vec2)

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
            # Ayx2[m,n] = -1*result

get_Axy2()


Axy1_np = Axy1.to_numpy()
del Axy1
gc.collect()
Axy1_complex = Axy1_np[:, :, 0] + 1j * Axy1_np[:, :, 1]
del Axy1_np
gc.collect()

def matvec_A1(un):
    Mx, My = un[:int(len(un)/2)], un[int(len(un)/2):]
    return np.concatenate([Axy1_complex @ My, -Axy1_complex @ Mx])

def matvec_A1H(v):
    vx, vy = v[:int(len(v)/2)], v[int(len(v)/2):]
    return np.concatenate([-Axy1_complex.conj().T @ vy, 
                            Axy1_complex.conj().T @ vx])

Axy2_np = Axy2.to_numpy()
del Axy2
gc.collect()
Axy2_complex = Axy2_np[:, :, 0] + 1j * Axy2_np[:, :, 1]
del Axy2_np
gc.collect()

def matvec_A2(un):
    Mx, My = un[:int(len(un)/2)], un[int(len(un)/2):]
    return np.concatenate([Axy2_complex @ My, -Axy2_complex @ Mx])

def matvec_A2H(v):
    vx, vy = v[:int(len(v)/2)], v[int(len(v)/2):]
    return np.concatenate([-Axy2_complex.conj().T @ vy, 
                            Axy2_complex.conj().T @ vx])

print("Done getting A1, A2, f1, f2.")

f1_abs = np.abs(f1)
f2_abs = np.abs(f2)

f1_scale = np.linalg.norm(f1_abs)
f2_scale = np.linalg.norm(f2_abs)

f1_abs_n = f1_abs / f1_scale
f2_abs_n = f2_abs / f2_scale

f1_abs = f1_abs_n
f2_abs = f2_abs_n
# eta1 = (np.linalg.norm(f1_abs_n**2))**2   # now order 1
# eta2 = (np.linalg.norm(f2_abs_n**2))**2 

eta1 = (np.linalg.norm(f1_abs**2))**2
eta2 = (np.linalg.norm(f2_abs**2))**2
print("eta:", eta1, eta2)
u_BP = matvec_A1H(f1_abs)
zetas = np.linspace(0.01, 10, 1000)
costs = []
for zeta in zetas:
    u_test = zeta * u_BP
    r1_test = np.abs(matvec_A1(u_test))**2 - f1_abs**2
    r2_test = np.abs(matvec_A2(u_test))**2 - f2_abs**2
    costs.append(eta1 * np.linalg.norm(r1_test)**2 + eta2 * np.linalg.norm(r2_test)**2)

zeta = zetas[np.argmin(costs)]
un = zeta * u_BP

tolerance_factor = 1e-7
error = 1e3
area_d = del_y*n_cols*del_x*n_rows

gn_previous = 0
gn = 0
vn = 0

alpha_n = 1
Nd = N_quad 
beta_n = 0
count = 0

# def line_search(un, vn, f1_abs, f2_abs, eta1, eta2):
#     alpha = 1.0
#     rho = 0.5      
#     max_iter = 50
    
#     r1 = np.abs(matvec_A1(un))**2 - f1_abs**2
#     r2 = np.abs(matvec_A2(un))**2 - f2_abs**2
#     c0 = eta1 * np.linalg.norm(r1)**2 + eta2 * np.linalg.norm(r2)**2
    
#     for _ in range(max_iter):
#         un_new = un + alpha * vn   
#         r1_new = np.abs(matvec_A1(un_new))**2 - f1_abs**2
#         r2_new = np.abs(matvec_A2(un_new))**2 - f2_abs**2
#         c_new = eta1 * np.linalg.norm(r1_new)**2 + eta2 * np.linalg.norm(r2_new)**2
        
#         if c_new < c0:
#             return alpha
#         alpha *= rho
    
#     return alpha
def line_search(un, vn, f1_abs, f2_abs, eta1, eta2, b_n_sq, del_sqr, del_x, del_y, n_rows, n_cols):
    alpha = 1.0
    rho = 0.5      
    max_iter = 50
    
    # Base Cost
    r1 = np.abs(matvec_A1(un))**2 - f1_abs**2
    r2 = np.abs(matvec_A2(un))**2 - f2_abs**2
    C_data = eta1 * np.linalg.norm(r1)**2 + eta2 * np.linalg.norm(r2)**2
    # C_MR is technically exactly 1.0 for 'un' at the current iteration by definition of b_n
    c0 = C_data * 1.0 
    
    for _ in range(max_iter):
        un_new = un + alpha * vn   
        
        # 1. Calculate new data mismatch
        r1_new = np.abs(matvec_A1(un_new))**2 - f1_abs**2
        r2_new = np.abs(matvec_A2(un_new))**2 - f2_abs**2
        C_data_new = eta1 * np.linalg.norm(r1_new)**2 + eta2 * np.linalg.norm(r2_new)**2
        
        # 2. Calculate new MR penalty using fixed b_n from the outer loop
        Mx_new = un_new[:N_quad].reshape(n_rows, n_cols)
        My_new = un_new[N_quad:].reshape(n_rows, n_cols)
        
        grad_y_x_new, grad_x_x_new = np.gradient(Mx_new, del_y, del_x)
        grad_y_y_new, grad_x_y_new = np.gradient(My_new, del_y, del_x)
        
        grad_mag_sq_new = np.abs(grad_x_x_new)**2 + np.abs(grad_y_x_new)**2 + np.abs(grad_x_y_new)**2 + np.abs(grad_y_y_new)**2
        
        # Area integral for C_MR
        C_MR_new = np.sum(b_n_sq * (grad_mag_sq_new + del_sqr)) * (del_x * del_y)
        
        # 3. Total new cost
        c_new = C_data_new * C_MR_new
        
        if c_new < c0:
            return alpha
        alpha *= rho
    
    return alpha
while error > tolerance_factor:
    r1 = (np.abs(matvec_A1(un)))**2 - f1_abs**2
    r2 = (np.abs(matvec_A2(un)))**2 - f2_abs**2
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
    print(b_n.shape)

    b_n_sq = b_n**2

    div_x_mx = np.gradient(b_n_sq * grad_x_x, del_x, axis=1)
    div_y_mx = np.gradient(b_n_sq * grad_y_x, del_y, axis=0)
    g_MR_x = -(div_x_mx + div_y_mx)  

    div_x_my = np.gradient(b_n_sq * grad_x_y, del_x, axis=1)
    div_y_my = np.gradient(b_n_sq * grad_y_y, del_y, axis=0)
    g_MR_y = -(div_x_my + div_y_my)
    g_MR = np.concatenate([g_MR_x.flatten(), g_MR_y.flatten()])
    term1 = matvec_A1H(2*eta1*r1 * matvec_A1(un))  
    term2 = matvec_A2H(2*eta2*r2 * matvec_A2(un))
    term3 = (c1 + c2)*g_MR

    gn_previous = gn
    gn = term1 + term2 + term3
    print("gn: ",np.linalg.norm(gn))

    g_flat = gn.conj().flatten()
    diff_flat = (gn - gn_previous).flatten()
    numerator = np.dot(g_flat, diff_flat)          
    denominator = np.linalg.norm(gn_previous)**2       
    if count != 0:
        beta_n = numerator / denominator
    else: beta_n = 0

    vn = gn + beta_n * vn
    print("vn value: ", np.linalg.norm(vn))
    un_previous = un
    alpha_n = line_search(un, vn, f1_abs, f2_abs, eta1, eta2, b_n_sq, del_sqr, del_x, del_y, n_rows, n_cols)
    print("alpha: ", alpha_n)
    un = un + alpha_n * vn 
    error = np.abs(np.linalg.norm(un-un_previous)/np.linalg.norm(un))
    count += 1
    print("error: ", error)

    if count > 100:
        print("Not Converging",error)
        break

print(error, count)

Mx_2d = un[:N_quad].reshape(n_rows, n_cols)
My_2d = un[N_quad:].reshape(n_rows, n_cols)

#plot_data = np.abs(Mx_2d)**2 + np.abs(My_2d)**2   
mag = np.sqrt(np.abs(Mx_2d)**2 + np.abs(My_2d)**2)
#plot_data_db = 20 * np.log10(mag / np.max(mag))
# Then clip to e.g., -40dB for a clean visual
plot_data = mag#np.clip(plot_data_db, -40, 0)
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
# Assuming R and d are defined
R = 81.0 # example radius
d = 0.005  # example depth
a = d / (R**2)

# Copy your coordinates to avoid modifying the original mesh data
parabolic_coords = reconstructed_coordinates.copy()

# Update the Z-axis (index 2) based on X (index 0) and Y (index 1)
x = parabolic_coords[:, 0]
y = parabolic_coords[:, 1]
parabolic_coords[:, 2] = a * (x**2 + y**2)
# Create mesh with new 3D parabolic coordinates
pv_mesh = pv.PolyData(parabolic_coords, cells)

# Add your SRM output data
pv_mesh.cell_data["Current Magnitude"] = cell_data

# Plot in 3D
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, scalars="Current Magnitude", cmap="jet")
plotter.show()
