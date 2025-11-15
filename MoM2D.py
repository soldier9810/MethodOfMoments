import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import meshio
from collections import defaultdict
import pyvista as pv
import time
from datetime import timedelta
from tqdm import tqdm

start_time = time.perf_counter()

c = 3e8
f = 48e8
wavelength = c/f
k = 2*np.pi/wavelength
R = np.linspace(0.01, 2, 500)
H0 = special.hankel1(0, k * R)
w = 2*np.pi*f
u = 4*np.pi*1e-7
epsilon = 8.85*1e-12

def electric_field(r):
    return np.array([1, 0, 0]) * np.exp(-1j * k * r[2])


nop = 12
############## 7 point
# gauss_points = np.array([[0.3333, 0.0597, 0.4701, 0.4701, 0.7974, 0.1013, 0.1013],
#                 [0.3333, 0.4701, 0.0597, 0.4701, 0.1013, 0.7974, 0.1013],
#                 [0.3333, 0.4701, 0.4701, 0.0597, 0.1013, 0.1013, 0.7974]])
# gauss_weights = np.array([0.225, 0.1323, 0.1323, 0.1323, 0.1259, 0.1259, 0.1259])

############## 4 point
if nop == 7:
    alpha = np.array([0.3333, 0.0597, 0.4701, 0.4701, 0.7974, 0.1013, 0.1013])
    beta  = np.array([0.3333, 0.4701, 0.0597, 0.4701, 0.1013, 0.7974, 0.1013])
    gamma = np.array([0.3333, 0.4701, 0.4701, 0.0597, 0.1013, 0.1013, 0.7974])
    w_gauss = np.array([0.225, 0.1323, 0.1323, 0.1323, 0.1259, 0.1259, 0.1259])
    
elif nop == 4:
    alpha = np.array([0.3333333, 0.6000000, 0.2000000, 0.2000000])
    beta  = np.array([0.3333333, 0.2000000, 0.6000000, 0.2000000])
    gamma = np.array([0.3333333, 0.2000000, 0.2000000, 0.6000000])
    w_gauss     = np.array([-0.56250000, 0.52083333, 0.52083333, 0.52083333])
############# 6 point
elif nop == 6:
    alpha = np.array([0.10810301, 0.44594849, 0.44594849, 0.81684757, 0.09157621, 0.09157621])
    beta  = np.array([0.44594849, 0.10810301, 0.44594849, 0.09157621, 0.81684757, 0.09157621])
    gamma = np.array([0.44594849, 0.44594849, 0.10810301, 0.09157621, 0.09157621, 0.81684757])
    w_gauss     = np.array([0.22338158, 0.22338158, 0.22338158, 0.10995174, 0.10995174, 0.10995174])
############# 3 point
elif nop == 3:
    alpha = np.array([0.66666667, 0.16666667, 0.16666667])
    beta  = np.array([0.16666667, 0.66666667, 0.16666667])
    gamma = np.array([0.16666667, 0.16666667, 0.66666667])
    w_gauss     = np.array([0.33333333, 0.33333333, 0.33333333])

elif nop == 16:
    alpha = np.array([0.081414823414554, 0.081414823414554, 0.837170353641812,
                    0.658861384496480, 0.658861384496480, 0.316277231295461,
                    0.316277231295461, 0.024180199784358, 0.024180199784358,
                    0.024180199784358, 0.310352451033785, 0.636502499121399,
                    0.636502499121399, 0.636502499121399, 0.053145049844816,
                    0.053145049844816])

    beta  = np.array([0.081414823414554, 0.837170353641812, 0.081414823414554,
                    0.316277231295461, 0.024180199784358, 0.658861384496480,
                    0.024180199784358, 0.658861384496480, 0.024180199784358,
                    0.316277231295461, 0.053145049844816, 0.053145049844816,
                    0.310352451033785, 0.310352451033785, 0.636502499121399,
                    0.636502499121399])

    gamma = 1.0 - alpha - beta

    w_gauss     = np.array([0.090817990382754, 0.090817990382754, 0.090817990382754,
                    0.082851075618374, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.025731066440455, 0.025731066440455,
                    0.025731066440455, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.082851075618374, 0.025731066440455,
                    0.025731066440455])

elif nop == 12:
    alpha = np.array([0.249286745170910, 0.249286745170910, 0.501426509658179,
                    0.063089014491502, 0.063089014491502, 0.873821971016996,
                    0.310352451033785, 0.636502499121399, 0.053145049844816,
                    0.310352451033785, 0.636502499121399, 0.053145049844816])

    beta  = np.array([0.249286745170910, 0.501426509658179, 0.249286745170910,
                    0.063089014491502, 0.873821971016996, 0.063089014491502,
                    0.636502499121399, 0.053145049844816, 0.310352451033785,
                    0.053145049844816, 0.310352451033785, 0.636502499121399])

    gamma = 1.0 - alpha - beta

    w_gauss     = np.array([0.116786275726379, 0.116786275726379, 0.116786275726379,
                    0.050844906370207, 0.050844906370207, 0.050844906370207,
                    0.082851075618374, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.082851075618374, 0.082851075618374])


gauss_points = np.array([alpha,
                beta,
                gamma])
gauss_weights = np.array(w_gauss)

#mesh = meshio.read(r"C:\Users\aryan\Downloads\keyKap-Body.msh")
mesh = meshio.read(r"C:\Users\aryan\Downloads\MoM_Test_Plane-TestPlane.msh")
# mesh = meshio.read(r"C:\Users\aryan\Downloads\plane.msh")
# mesh = meshio.read(r"C:\Users\aryan\Downloads\plane50points.msh")
#mesh = meshio.read(r"C:\Users\aryan\Downloads\sphere30points.msh")
# mesh = meshio.read(r"C:\Users\aryan\Downloads\sphere40points.msh")
points = mesh.points
triangles = mesh.cells_dict["triangle"]
print(len(triangles))
for i in range(0,len(points)):
    points[i] = (np.array(points[i])/1e6)
for i in range(0,len(triangles)):
    triangles[i] = np.array(triangles[i])

min_corner = np.min(points, axis=0)
max_corner = np.max(points, axis=0)
bbox_center = (min_corner + max_corner) / 2.0
points -= bbox_center

avg_side_length = 0
for tri in points[triangles]:
    s1 = np.linalg.norm(tri[0] - tri[1])
    s2 = np.linalg.norm(tri[0] - tri[2])
    s3 = np.linalg.norm(tri[2] - tri[1])
    avg_side_length += (s1 + s2 + s3) / 3

avg_side_length = avg_side_length / len(triangles)

ncap_global = []

for i, tri in enumerate(triangles):
    v = points[tri]
    normal = np.cross(v[1]-v[0], v[2]-v[0])
    centroid = (v[0] + v[1] + v[2])/3
    if np.dot(normal, centroid) < 0:
        triangles[i] = triangles[i][::-1]  # reverse winding

    v = points[triangles[i]]
    normal = np.cross(v[1]-v[0], v[2]-v[0])
    normal /= np.linalg.norm(normal)
    ncap_global.append(normal)

ncap_global = np.array(ncap_global)


triangle_area = np.zeros((len(triangles)))
triangle_centroid = np.zeros((len(triangles), 3))

def ensure_outward(v, ref_point):
    if np.dot(v, ref_point) < 0:
        v = -v
    return v

def get_area(triangle):
    tri = points[triangle]
    side1 = tri[1] - tri[0]
    side2 = tri[2] - tri[0]
    area = 0.5 * np.linalg.norm(np.cross(side1, side2))
    centroid = (tri[0] + tri[1] + tri[2])/3
    return area, centroid


for i in range(0,len(triangles)):
    triangle_area[i], triangle_centroid[i] = get_area(triangles[i])

def func(rp, rq, rp_type, rq_type, rp_free, rq_free):
    global w, u, epsilon, k
    distance = np.linalg.norm(rp - rq)
    rho_rp = ((-1)**rp_type)*(rp - rp_free)
    rho_rq = ((-1)**rq_type)*(rq - rq_free)
    first_term = (1j*w*u/4)*np.dot(rho_rp,rho_rq)
    second_term = (1j/(w*epsilon))*((-1)**rp_type)*((-1)**rq_type)
    return (first_term - second_term)*(np.e**((-1j)*k*distance))/distance

def func_vectorized(rp, rq, rp_type, rq_type, rp_free, rq_free):
    global w, u, epsilon, k

    # Ensure arrays can broadcast properly
    rho_rp = ((-1)**rp_type) * (rp - rp_free)   # shape (1,7,3)
    rho_rq = ((-1)**rq_type) * (rq - rq_free)   # shape (7,1,3)

    # Compute pairwise distances |rp - rq|
    R = rp - rq                                 # shape (7,7,3)
    distance = np.linalg.norm(R, axis=-1)       # shape (7,7)

    # Dot product between rho_rp[i,:] and rho_rq[j,:] for all i,j
    dot_rho = np.einsum('ijk,ijk->ij',
                        rho_rp + np.zeros_like(rq),  # broadcast (7,7,3)
                        rho_rq + np.zeros_like(rp))  # broadcast (7,7,3)
    # Alternatively: use broadcasting trick:
    # dot_rho = np.sum(rho_rp[:, None, :] * rho_rq[None, :, :], axis=-1)

    first_term = (1j * w * u / 4) * dot_rho
    second_term = (1j / (w * epsilon)) * ((-1)**rp_type) * ((-1)**rq_type)

    kernel = (first_term - second_term) * np.exp(-1j * k * distance) / distance
    return kernel


def func_singularity(rp, rq, rp_type, rq_type, rp_free, rq_free, tout, tin, inner):
    ################ in, out
    global w, u, epsilon, k
    if inner:
        distance = np.linalg.norm(rp - rq)
        rho_rp = ((-1)**rp_type)*(rp - rp_free)
        rho_rq = ((-1)**rq_type)*(rq - rq_free)
        first_term = (1j*w*u/4)*np.dot(rho_rp,rho_rq) - (1j/(w*epsilon))*((-1)**rp_type)*((-1)**rq_type)
        if distance == 0:
            first_term *= (-1j*k)
        else:
            first_term *= (np.e**(-1j*k*distance) - 1)/distance

        return first_term
    else:
        tri = points[triangles[tin]]
        side11 = tri[0] - tri[1]
        side12 = tri[0] - tri[2]
        ncap = np.cross(side11, side12)
        ncap /= np.linalg.norm(ncap)

        # rho_rq = ((-1)**rq_type)*(rq - rq_free)
        # I1, I2 = get_I1I2(tin, rq, ncap)
        # third_term = I2*(1j/(w*epsilon))*((-1)**rp_type)*((-1)**rq_type)
        # second_term = np.dot(rho_rq,((-1)**rp_type)*(I1 + (rq - rp_free)*I2))
        # return (second_term - third_term)/triangle_area[tin]

        rho_rp = ((-1)**rp_type)*(rp - rp_free)
        I1, I2 = get_I1I2(tin, rp, ncap)
        third_term = I2*(1j/(w*epsilon))*((-1)**rp_type)*((-1)**rq_type)
        second_term = np.dot(rho_rp,((-1)**rq_type)*(I1 + (rp - rq_free)*I2))
        return (second_term - third_term)/triangle_area[tin]
    

def get_component_in_plane(ncap, r):
    return r - ncap*np.dot(ncap,r)


def get_I1I2(tin, obv_point, ncap):
    tri = points[triangles[tin]]
    eps = 1e-12
    #rho_prime = get_component_in_plane(ncap, source_point)
    I1 = np.array([0,0,0], dtype = complex)
    I2 = 0
    for i in range(0,3):
        if i == 2:
            p1 = tri[0]
            p2 = tri[2]
        else:
            p1 = tri[i]
            p2 = tri[(i+1)%3]
        lcap = (p2 - p1)
        lcap = lcap/np.linalg.norm(lcap)
        ucap = np.cross(lcap,ncap)      
        obv_point = obv_point + ucap*1e-12
        rho = get_component_in_plane(ncap, obv_point)
        rho_p1 = get_component_in_plane(ncap, p1)
        rho_p2 = get_component_in_plane(ncap, p2)
        l_plus = np.dot((rho_p2 - rho),lcap)
        l_minus = np.dot((rho_p1 - rho),lcap)
        p0 = abs(np.dot(ucap, rho_p1 - rho))
        p_plus = np.linalg.norm(rho_p2 - rho)
        p_minus = np.linalg.norm(rho_p1 - rho)
        p0_cap = ((rho_p1 - rho) - l_minus * lcap)
        p0_cap /= p0
        d = abs(np.dot(ncap, obv_point - p1))
        R0 = np.sqrt(p0**2 + d**2)
        Rplus = np.sqrt(p_plus**2 + d**2)
        Rminus = np.sqrt(p_minus**2 + d**2)
        common_term = np.log((Rplus + l_plus + eps) / (Rminus + l_minus + eps))
        value = ((R0**2)*common_term) + l_plus*Rplus - l_minus*Rminus
        I1 += value*ucap
        tan_term1 = np.arctan(p0*l_plus/(R0**2 + d*Rplus + eps))
        tan_term2 = np.arctan(p0*l_minus/(R0**2 + d*Rminus + eps))
        value = (p0*common_term) - d*(tan_term1 - tan_term2)
        value *= np.dot(p0_cap,ucap)
        I2 += value

    I1 = I1/2
    return I1, I2
def get_I1I2_vectorized(tin, obv_point, ncap):
    tri = points[triangles[tin]]

    I1 = np.zeros(3, dtype=complex)
    I2 = 0.0

    eps = 1e-12

    for i in range(3):
        p1 = tri[i]
        p2 = tri[(i+1)%3]

        # Edge direction and perpendicular in-plane
        lcap = (p2 - p1)
        lcap /= (np.linalg.norm(lcap) + eps)
        ucap = np.cross(lcap, ncap)

        # Tiny nudge to avoid exact-edge singularity (local copy only!)
        rho_obs = obv_point + eps * ucap

        # Projections into plane
        rho     = rho_obs - ncap*np.dot(ncap, rho_obs)
        rho_p1  = p1      - ncap*np.dot(ncap, p1)
        rho_p2  = p2      - ncap*np.dot(ncap, p2)

        # l_plus & l_minus (signed)
        l_plus  = np.dot((rho_p2 - rho), lcap)
        l_minus = np.dot((rho_p1 - rho), lcap)

        # p0 & P_plus/P_minus
        P0  = abs(np.dot((rho_p1 - rho), ucap))  # scalar
        Ppl = np.linalg.norm(rho_p2 - rho)
        Pmn = np.linalg.norm(rho_p1 - rho)

        # normal distance
        d = abs(np.dot(ncap, rho_obs - p1))

        # Distances
        R0    = np.sqrt(P0*P0 + d*d)
        Rplus = np.sqrt(Ppl*Ppl + d*d)
        Rminus= np.sqrt(Pmn*Pmn + d*d)

        # Safe log
        ratio = (Rplus + l_plus + eps) / (Rminus + l_minus + eps)
        ratio = np.clip(ratio, eps, None)

        # I1 contribution
        val1 = (R0*R0)*np.log(ratio) + l_plus*Rplus - l_minus*Rminus
        I1 += val1 * ucap

        # atan terms (safe denom)
        den1 = (R0*R0) + d*Rplus + eps
        den2 = (R0*R0) + d*Rminus + eps
        t1 = np.arctan((P0*l_plus )/den1)
        t2 = np.arctan((P0*l_minus)/den2)

        # I2 contribution (no vector projection!)
        val2 = P0*(np.log(ratio) - d*(t1 - t2))
        I2 += val2

    return 0.5*I1, I2


def func_singularity_vectorized(Rp, Rq, rp_type, rq_type,
                                rp_free, rq_free, tout, tin):
    """
    Rp: shape (7,7,3) observation/test points
    Rq: shape (7,7,3) source points
    """

    global w, u, epsilon, k

    # Compute analytic integrals once per triangle pair
    I1_vec, I2_sca = get_I1I2_vectorized(tin, triangle_centroid[tout], ncap_global[tin])
    # I1 is (3,), I2 is scalar

    # Vectorize geometric quantities
    rho_rq = ((-1)**rq_type) * (Rq - rq_free)
    rho_rp = ((-1)**rp_type) * (Rp - rp_free)

    diff = Rp - Rq
    dist = np.linalg.norm(diff, axis=-1) + 1e-16

    # First term (inner singular expansion)
    termA = (1j * w * u / 4) * np.einsum('ijk,ijk->ij', rho_rp, rho_rq)

    # Analytical expansion factor when dist → 0
    kernel_sing = (np.exp(-1j * k * dist) - 1.0) / (dist + 1e-16)

    term_finite = termA * kernel_sing

    # Second term from analytic I2
    termC = (1j / (w * epsilon)) * ((-1)**rp_type)*((-1)**rq_type)
    term_edge = termC * I2_sca

    # Add vector component from I1
    contrib_I1 = np.einsum('k,ijk->ij', I1_vec, rho_rq)  # dot(I1, rho_rq)

    term_edge += ((-1)**rp_type) * contrib_I1

    # Normalize by triangle area (only once)
    term_edge /= triangle_area[tin]

    return term_finite + term_edge


def integration3D(v1,v2,v3,func):
    result = 0
    for i in range(0,len(gauss_weights)):
        location = gauss_points[0][i]*v1 + gauss_points[1][i]*v2 + gauss_points[2][i]*v3
        result += func(location)*gauss_weights[i]
    return result #area*result          ####### REMOVING AREA SINCE CANCELS OUT IN FINAL EQUATION

def double_integration(vout1, vout2, vout3, vin1, vin2, vin3, rp_type, rq_type, rp_free, rq_free, tout, tin):
    ######## REMOVING AREA FROM THIS FUNCTION SINCE IT CANCELS OUT IN THE FINAL EQUATION
    final_result = 0
    delta_rc = np.linalg.norm(triangle_centroid[tin] - triangle_centroid[tout])
    if delta_rc > avg_side_length*0.001:
        final_result = 0
        for i in range(0,len(gauss_weights)):
            location_out = gauss_points[0][i]*vout1 + gauss_points[1][i]*vout2 + gauss_points[2][i]*vout3
            result = 0
            for j in range(0,len(gauss_weights)):
                location_in = gauss_points[0][j]*vin1 + gauss_points[1][j]*vin2 + gauss_points[2][j]*vin3
                result += func(location_out, location_in, rp_type, rq_type, rp_free, rq_free)*gauss_weights[j]

            final_result += result*gauss_weights[i]

    else:
        final_result = 0
        for i in range(0,len(gauss_weights)):
            location_out = gauss_points[0][i]*vout1 + gauss_points[1][i]*vout2 + gauss_points[2][i]*vout3
            result = 0
            for j in range(0,len(gauss_weights)):
                location_in = gauss_points[0][j]*vin1 + gauss_points[1][j]*vin2 + gauss_points[2][j]*vin3
                result += func_singularity(location_out, location_in, rp_type, rq_type, rp_free, rq_free, tout, tin, 1)*gauss_weights[j]
            result += func_singularity(location_out, location_in, rp_type, rq_type, rp_free, rq_free, tout, tin, 0)
            final_result += result*gauss_weights[i]

    return final_result



def double_integration_vectorized(vout1, vout2, vout3,
                                  vin1, vin2, vin3, rp_type, rq_type, rp_free, rq_free, tout, tin):
    ######## REMOVING AREA FROM THIS FUNCTION SINCE IT CANCELS OUT IN THE FINAL EQUATION
    # 7-point Dunavant barycentric coordinates and weights
    integral = 0
    set1 = set(triangles[tin])
    set2 = set(triangles[tout])
    shared = len(set1.intersection(set2))
    # centroid_out = (vout1 + vout2 + vout3)/3
    # centroid_in = (vin1 + vin2 + vin3)/3
    # delta_rc = np.linalg.norm(centroid_in - centroid_out)
    r_out = (alpha[:,None]*vout1 +
            beta[:,None]*vout2 +
            gamma[:,None]*vout3)
    r_in  = (alpha[:,None]*vin1 +
            beta[:,None]*vin2 +
            gamma[:,None]*vin3)

    R_out = r_out[:,None,:]               
    R_in  = r_in[None,:,:]
    # if delta_rc > 2*avg_side_length:
    # if tout != tin:
    if shared < 2:
        F = func_vectorized(R_out, R_in, rp_type, rq_type, rp_free, rq_free)
        W = np.outer(w_gauss, w_gauss)
        integral = np.sum(F * W)
    else:
        final_result = 0
        for i in range(0,len(gauss_weights)):
            location_out = gauss_points[0][i]*vout1 + gauss_points[1][i]*vout2 + gauss_points[2][i]*vout3
            result = 0
            for j in range(0,len(gauss_weights)):
                location_in = gauss_points[0][j]*vin1 + gauss_points[1][j]*vin2 + gauss_points[2][j]*vin3
                result += func_singularity(location_out, location_in, rp_type, rq_type, rp_free, rq_free, tout, tin, 1)*gauss_weights[j]
            result += func_singularity(location_out, location_in, rp_type, rq_type, rp_free, rq_free, tout, tin, 0)
            final_result += result*gauss_weights[i]
        integral = final_result
        # F = func_singularity_vectorized(R_out, R_in, rp_type, rq_type, rp_free, rq_free, tout, tin)
        # W = np.outer(w_gauss, w_gauss)
        # integral = np.sum(F * W) 
    return integral


edge_dict = defaultdict(list)

for tri_id, tri in enumerate(triangles):
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

    tri1 = triangles[t1]
    tri2 = triangles[t2]
    tri1_pts = points[tri1]
    tri2_pts = points[tri2]
    side11 = tri1_pts[0] - tri1_pts[1]
    side12 = tri1_pts[0] - tri1_pts[2]
    area1 = 0.5 * np.linalg.norm(np.cross(side11, side12))
    side21 = tri2_pts[0] - tri2_pts[1]
    side22 = tri2_pts[0] - tri2_pts[2]
    area2 = 0.5 * np.linalg.norm(np.cross(side21, side22))
    edge_length = np.linalg.norm(points[n2] - points[n1])
    pos_tri, neg_tri = t1, t2
    pos_area, neg_area = area1, area2
    free_pos = points[list(set(triangles[pos_tri]) - set(edge))[0]]
    free_neg = points[list(set(triangles[neg_tri]) - set(edge))[0]]
    common1 = points[n1]
    common2 = points[n2]
    common_points = [common1, common2]
    all_basis[count] = [free_neg, free_pos, edge_length, common_points, pos_tri, neg_tri, pos_area, neg_area]
    ######################  0        1           2              3          4        5          6       7
    count = count + 1

def get_excitation(vminus, vplus, common1, common2, edge_length):

    sum_minus = 0
    sum_plus = 0

    for i in range(0,len(gauss_weights)):
        location_minus = gauss_points[0][i]*vminus + gauss_points[1][i]*common1 + gauss_points[2][i]*common2
        rho_minus = location_minus - vminus
        func_minus = np.dot(rho_minus,electric_field(location_minus))
        sum_minus += func_minus*gauss_weights[i]

        location_plus = gauss_points[0][i]*vplus + gauss_points[1][i]*common1 + gauss_points[2][i]*common2
        rho_plus = vplus - location_plus
        func_plus = np.dot(rho_plus,electric_field(location_plus))
        sum_plus += func_plus*gauss_weights[i]

    return (edge_length/2)*(sum_plus + sum_minus)

N = len(all_basis)
print(N)
Z = np.zeros((N, N), dtype=complex)
I = np.zeros((N,1), dtype=complex)
for m in tqdm(range(N)):
    mth_basis = all_basis[m]
    Lm = mth_basis[2]
    I[m] = get_excitation(mth_basis[0], mth_basis[1], mth_basis[3][0], mth_basis[3][1], Lm)
    for n in range(0,N):
        nth_basis = all_basis[n]       
        Ln = nth_basis[2]                                                                                                                               #rp_type, rq_type, rp_free, rq_free
        Imneg_nneg = double_integration_vectorized(mth_basis[0], mth_basis[3][0], mth_basis[3][1], nth_basis[0], nth_basis[3][0], nth_basis[3][1], 0, 0, mth_basis[0], nth_basis[0], mth_basis[5], nth_basis[5])
        Imneg_npos = double_integration_vectorized(mth_basis[0], mth_basis[3][0], mth_basis[3][1], nth_basis[1], nth_basis[3][0], nth_basis[3][1], 0, 1, mth_basis[0], nth_basis[1], mth_basis[5], nth_basis[4])
        Impos_nneg = double_integration_vectorized(mth_basis[1], mth_basis[3][0], mth_basis[3][1], nth_basis[0], nth_basis[3][0], nth_basis[3][1], 1, 0, mth_basis[1], nth_basis[0], mth_basis[4], nth_basis[5])
        Impos_npos = double_integration_vectorized(mth_basis[1], mth_basis[3][0], mth_basis[3][1], nth_basis[1], nth_basis[3][0], nth_basis[3][1], 1, 1, mth_basis[1], nth_basis[1], mth_basis[4], nth_basis[4])
        # # Imneg_nneg = double_integration(mth_basis[0], mth_basis[3][0], mth_basis[3][1], nth_basis[0], nth_basis[3][0], nth_basis[3][1], 0, 0, nth_basis[0], mth_basis[0], mth_basis[5], nth_basis[5])
        # # Imneg_npos = double_integration(mth_basis[0], mth_basis[3][0], mth_basis[3][1], nth_basis[1], nth_basis[3][0], nth_basis[3][1], 0, 1, nth_basis[1], mth_basis[0], mth_basis[5], nth_basis[4])
        # # Impos_nneg = double_integration(mth_basis[1], mth_basis[3][0], mth_basis[3][1], nth_basis[0], nth_basis[3][0], nth_basis[3][1], 1, 0, nth_basis[0], mth_basis[1], mth_basis[4], nth_basis[5])
        # # Impos_npos = double_integration(mth_basis[1], mth_basis[3][0], mth_basis[3][1], nth_basis[1], nth_basis[3][0], nth_basis[3][1], 1, 1, nth_basis[1], mth_basis[1], mth_basis[4], nth_basis[4])
        Z[m][n] = (Lm*Ln/(4*np.pi))*(Imneg_nneg + Imneg_npos + Impos_nneg + Impos_npos)

print("Done calculating Z")
np.save("Z_planeGauss16freq48.npy", Z)
# Z = np.load("Z_planeGauss3.npy")
Z = np.round(Z,12)
I = np.round(I,12)
coeff = np.linalg.solve(Z, I)

for i in tqdm(range(N), desc="Assembling colors"):
    ith_basis = all_basis[i]
    value = coeff[i]*ith_basis[2]/2

    pos_tri = points[triangles[ith_basis[4]]]
    pos_centroid = (pos_tri[0] + pos_tri[1] + pos_tri[2])/3
    pos_vec = ith_basis[1] - pos_centroid
    organized_triangles[ith_basis[4]] += (value/ith_basis[6])*pos_vec

    neg_tri = points[triangles[ith_basis[5]]]
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
    tri = points[triangles[tri_idx]]
    start_index = len(vertices)
    vertices.extend(tri)
    faces.extend([3, start_index, start_index + 1, start_index + 2])
    scalars.append(np.linalg.norm(value))
vertices = np.array(vertices)
mesh = pv.PolyData(vertices, faces)
mesh.cell_data["Magnitude"] = np.array(scalars)

mag = np.array(scalars)
vmin, vmax = 0, mag.max()  # or fixed global range
mesh.save("mesh_planeGauss16freq48.vtk")
# plotter = pv.Plotter()
# plotter.add_mesh(
#     mesh,
#     scalars="Magnitude",
#     cmap="turbo",
#     clim=[vmin, vmax],
#     show_edges=True,
#     edge_color='black',
#     smooth_shading=True,
#     lighting=True
# )
# plotter.add_scalar_bar(title="|Current| (A/m)")
# plotter.show_bounds(grid='front', location='outer', all_edges=True)
# plotter.enable_eye_dome_lighting()
# plotter.show()

# vertices = np.array(vertices)

# mesh = pv.PolyData(vertices, faces)

# mesh.cell_data["Magnitude"] = np.array(scalars)

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, scalars="Magnitude", cmap="coolwarm", show_edges=False)
# plotter.add_scalar_bar(title="|Current| (A/m)")
# plotter.show_axes()
# plotter.show_bounds(grid='front', location='outer', all_edges=True)
# plotter.show()

def far_field(rhat, m, r):
    mth_basis = all_basis[m]
    result_neg = np.array([0,0,0], dtype = complex)
    result_pos = np.array([0,0,0], dtype = complex)
    for i in range(0,len(gauss_weights)):
        location_neg = gauss_points[0][i]*mth_basis[0] + gauss_points[1][i]*mth_basis[3][0] + gauss_points[2][i]*mth_basis[3][1]
        rho_neg = location_neg - mth_basis[0]
        result_neg += gauss_weights[i]*rho_neg*(np.e**(1j*k*np.dot(rhat, rho_neg)))

        location_pos = gauss_points[0][i]*mth_basis[1] + gauss_points[1][i]*mth_basis[3][0] + gauss_points[2][i]*mth_basis[3][1]
        rho_pos = mth_basis[1] - location_pos
        result_pos += gauss_weights[i]*rho_pos*(np.e**(1j*k*np.dot(rhat, rho_pos)))

    return (-1j*w*u*np.e**(-1j*k*r))*mth_basis[2]*coeff[m]*(result_neg + result_pos)/(8*np.pi*r)


observations = 1000
observation_values = {}
theta = np.linspace(-np.pi, np.pi, num = observations)
#theta = np.linspace(-2*np.pi/3, 2*np.pi/3, num = observations)
r = 1
phi = 0
for angles in tqdm(theta):
    rhat = np.array([np.sin(angles)*np.cos(phi), np.sin(angles)*np.sin(phi), np.cos(angles)])
    observation_values[angles] = 0
    for m in range(0,N):
        observation_values[angles] += far_field(rhat, m, r)


angles = np.array(list(observation_values.keys()))
E_vectors = np.array(list(observation_values.values()))
E_magnitude = np.linalg.norm(E_vectors, axis=1)

eps = 1e-16

E_mag = np.abs(E_magnitude)

E_mag_norm = E_mag / (np.max(E_mag) + eps)

E_dB = 20.0 * np.log10(E_mag_norm + eps)

E_dB_clipped = np.clip(E_dB, a_min=-60.0, a_max=None)

plt.figure(figsize=(7,5))
plt.plot(np.degrees(angles), E_dB_clipped)
plt.xlabel("Theta (degrees)")
plt.ylabel("Normalized |E(θ)| (dB)")
plt.title("Far-Field Radiation Pattern")
plt.grid(True)
plt.tight_layout()
plt.savefig("normalized_planeGauss16freq48.png", dpi=200)
# plt.show()

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, E_mag_norm)   # values between 0 and 1
ax.set_title("Normalized Far-Field Pattern (Polar)")
plt.tight_layout()
plt.savefig("polar_planeGauss16freq48.png", dpi=200)
# plt.show()

# plt.figure(figsize=(7,5))
# plt.plot(np.degrees(angles), 20*np.log10(E_magnitude))
# plt.xlabel("Theta (degrees)")
# plt.ylabel("Normalized |E(θ)| (dB)")
# plt.title("Far-Field Radiation Pattern")
# plt.grid(True)
# plt.savefig("normalized_plane3.png")
# plt.show()


# plt.figure(figsize=(6,6))
# ax = plt.subplot(111, polar=True)
# ax.plot(angles, np.abs(E_magnitude)/np.max(E_magnitude))
# ax.set_title("Normalized Far-Field Pattern (Polar)")
# plt.savefig("polar_plane3.png")
# plt.show()


duration = timedelta(seconds=time.perf_counter()-start_time)
print(duration)