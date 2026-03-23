"""
Gmsh Quadrilateral Mesh Generator
Generate structured/unstructured quad meshes instead of triangular meshes
"""

import gmsh
import sys
import numpy as np


def create_structured_quad_mesh(output_filename, nx, ny, geometry_params=None):
    """
    Create a STRUCTURED quadrilateral mesh (perfect grid of squares)
    
    Parameters:
    -----------
    output_filename : str
        Output mesh file name
    nx : int
        Number of divisions in X direction
    ny : int
        Number of divisions in Y direction
    geometry_params : dict
        Geometry dimensions
    """
    
    if geometry_params is None:
        geometry_params = {
            'width': 1.0,
            'height': 1.0,
            'center': [0, 0, 0]
        }
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    gmsh.model.add("quad_plane")
    
    # Extract parameters
    width = geometry_params['width']
    height = geometry_params['height']
    cx, cy, cz = geometry_params['center']
    
    # Corner coordinates
    x_min = cx - width/2
    x_max = cx + width/2
    y_min = cy - height/2
    y_max = cy + height/2
    z = cz
    
    # Create corner points with TRANSFINITE meshing constraint
    # The lc parameter is ignored for transfinite meshing
    p1 = gmsh.model.geo.addPoint(x_min, y_min, z)
    p2 = gmsh.model.geo.addPoint(x_max, y_min, z)
    p3 = gmsh.model.geo.addPoint(x_max, y_max, z)
    p4 = gmsh.model.geo.addPoint(x_min, y_max, z)
    
    # Create lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    # Create curve loop and surface
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # CRITICAL: Set transfinite constraints for structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)  # Bottom edge
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)  # Right edge
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)  # Top edge
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)  # Left edge
    
    # Make surface transfinite (structured quad mesh)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    
    # Force quadrilateral meshing (not triangles)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    
    gmsh.model.geo.synchronize()
    
    # Mesh generation settings
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Force all quads
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Get statistics
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(2)
    
    num_nodes = len(nodes[0])
    num_quads = 0
    num_tris = 0
    
    for i, elem_type in enumerate(elements[0]):
        if elem_type == 3:  # Quadrilateral
            num_quads = len(elements[1][i])
        elif elem_type == 2:  # Triangle
            num_tris = len(elements[1][i])
    
    print(f"\n{'='*60}")
    print(f"Structured Quad Mesh: {output_filename}")
    print(f"{'='*60}")
    print(f"Geometry:")
    print(f"  Width:  {width} m")
    print(f"  Height: {height} m")
    print(f"  Divisions: {nx} × {ny}")
    print(f"\nMesh:")
    print(f"  Nodes:        {num_nodes}")
    print(f"  Quadrilaterals: {num_quads}")
    print(f"  Triangles:    {num_tris}")
    print(f"  Expected quads: {nx * ny}")
    print(f"{'='*60}\n")
    
    # Save
    gmsh.write(output_filename)
    
    if '-gui' in sys.argv:
        gmsh.fltk.run()
    
    gmsh.finalize()
    
    return num_nodes, num_quads


def create_unstructured_quad_mesh(output_filename, characteristic_length, 
                                   geometry_params=None, recombine_algorithm=1):
    """
    Create an UNSTRUCTURED quadrilateral mesh (irregular quads)
    
    Parameters:
    -----------
    output_filename : str
        Output mesh file name
    characteristic_length : float
        Target element size
    geometry_params : dict
        Geometry dimensions
    recombine_algorithm : int
        0 = simple (default)
        1 = blossom (better quality, slower)
        2 = blossom with split
        3 = full quad
    """
    
    if geometry_params is None:
        geometry_params = {
            'width': 1.0,
            'height': 1.0,
            'center': [0, 0, 0]
        }
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    gmsh.model.add("quad_plane_unstructured")
    
    # Extract parameters
    width = geometry_params['width']
    height = geometry_params['height']
    cx, cy, cz = geometry_params['center']
    
    # Corner coordinates
    x_min = cx - width/2
    x_max = cx + width/2
    y_min = cy - height/2
    y_max = cy + height/2
    z = cz
    
    # Create geometry
    p1 = gmsh.model.geo.addPoint(x_min, y_min, z, characteristic_length)
    p2 = gmsh.model.geo.addPoint(x_max, y_min, z, characteristic_length)
    p3 = gmsh.model.geo.addPoint(x_max, y_max, z, characteristic_length)
    p4 = gmsh.model.geo.addPoint(x_min, y_max, z, characteristic_length)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # Force recombination to quads
    gmsh.model.geo.mesh.setRecombine(2, surface)
    
    gmsh.model.geo.synchronize()
    
    # Meshing options
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine all triangular meshes
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recombine_algorithm)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    
    # Mesh quality optimization
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    # Mesh size control
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length * 0.8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length * 1.2)
    
    # Generate
    gmsh.model.mesh.generate(2)
    
    # Get statistics
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(2)
    
    num_nodes = len(nodes[0])
    num_quads = 0
    num_tris = 0
    
    for i, elem_type in enumerate(elements[0]):
        if elem_type == 3:  # Quadrilateral
            num_quads = len(elements[1][i])
        elif elem_type == 2:  # Triangle
            num_tris = len(elements[1][i])
    
    print(f"\n{'='*60}")
    print(f"Unstructured Quad Mesh: {output_filename}")
    print(f"{'='*60}")
    print(f"Geometry:")
    print(f"  Width:  {width} m")
    print(f"  Height: {height} m")
    print(f"  Target element size: {characteristic_length} m")
    print(f"\nMesh:")
    print(f"  Nodes:          {num_nodes}")
    print(f"  Quadrilaterals: {num_quads}")
    print(f"  Triangles:      {num_tris}")
    
    if num_tris > 0:
        print(f"  ⚠️  Warning: {num_tris} triangles present (not fully quad)")
        print(f"     Try different recombination algorithm or refine mesh")
    else:
        print(f"  ✓  Pure quad mesh!")
    
    print(f"{'='*60}\n")
    
    gmsh.write(output_filename)
    
    if '-gui' in sys.argv:
        gmsh.fltk.run()
    
    gmsh.finalize()
    
    return num_nodes, num_quads


def create_quad_mesh_from_step(step_filename, output_filename, 
                                characteristic_length, method='unstructured'):
    """
    Create quadrilateral mesh from STEP file
    
    Parameters:
    -----------
    step_filename : str
        Input STEP file
    output_filename : str
        Output mesh file
    characteristic_length : float
        Target element size
    method : str
        'unstructured' or 'structured' (if geometry allows)
    """
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Import STEP
    gmsh.model.occ.importShapes(step_filename)
    gmsh.model.occ.synchronize()
    
    # Get all surfaces
    surfaces = gmsh.model.getEntities(2)
    
    # Apply recombination to all surfaces
    for surface in surfaces:
        gmsh.model.mesh.setRecombine(surface[0], surface[1])
    
    # Mesh settings for quads
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    
    # Size control
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length * 1.5)
    
    # Quality optimization
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    # Generate
    gmsh.model.mesh.generate(2)
    
    # Statistics
    elements = gmsh.model.mesh.getElements(2)
    num_quads = 0
    num_tris = 0
    
    for i, elem_type in enumerate(elements[0]):
        if elem_type == 3:
            num_quads = len(elements[1][i])
        elif elem_type == 2:
            num_tris = len(elements[1][i])
    
    print(f"\nQuad mesh from STEP:")
    print(f"  Quadrilaterals: {num_quads}")
    print(f"  Triangles:      {num_tris}")
    
    gmsh.write(output_filename)
    
    if '-gui' in sys.argv:
        gmsh.fltk.run()
    
    gmsh.finalize()
    
    return num_quads, num_tris


def create_quad_mesh_suite(wavelength=1.0, divisions_list=[5, 10, 15, 20]):
    """
    Create suite of structured quad meshes with varying density
    
    Parameters:
    -----------
    wavelength : float
        Wavelength in meters
    divisions_list : list
        List of number of divisions per wavelength
    """
    
    geometry = {
        'width': 1.0 * wavelength,
        'height': 1.0 * wavelength,
        'center': [0, 0, 0]
    }
    
    print(f"\n{'='*80}")
    print(f"Creating QUAD mesh suite for wavelength = {wavelength} m")
    print(f"Geometry: {geometry['width']} m × {geometry['height']} m")
    print(f"{'='*80}\n")
    
    mesh_info = []
    
    for divisions in divisions_list:
        filename = f"plane_quad_{divisions}x{divisions}.msh"
        
        print(f"Generating {divisions}×{divisions} structured quad mesh...")
        num_nodes, num_quads = create_structured_quad_mesh(
            filename, 
            nx=divisions, 
            ny=divisions, 
            geometry_params=geometry
        )
        
        mesh_info.append({
            'filename': filename,
            'divisions': divisions,
            'nodes': num_nodes,
            'quads': num_quads
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("QUAD MESH SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Filename':<30} {'Divisions':<12} {'Nodes':<10} {'Quads':<10}")
    print(f"{'-'*80}")
    for info in mesh_info:
        div = info['divisions']
        print(f"{info['filename']:<30} {div}×{div:<10} {info['nodes']:<10} {info['quads']:<10}")
    print(f"{'='*80}\n")
    
    return mesh_info


def compare_tri_vs_quad_mesh(characteristic_length=0.1, geometry_params=None):
    """
    Generate both triangular and quad meshes for comparison
    """
    
    if geometry_params is None:
        geometry_params = {
            'width': 1.0,
            'height': 1.0,
            'center': [0, 0, 0]
        }
    
    print("\n" + "="*80)
    print("TRIANGLE vs QUAD MESH COMPARISON")
    print("="*80 + "\n")
    
    # Generate triangular mesh
    print("Generating TRIANGULAR mesh...")
    gmsh.initialize()
    gmsh.model.add("tri_plane")
    
    width = geometry_params['width']
    height = geometry_params['height']
    cx, cy, cz = geometry_params['center']
    
    x_min = cx - width/2
    x_max = cx + width/2
    y_min = cy - height/2
    y_max = cy + height/2
    
    p1 = gmsh.model.geo.addPoint(x_min, y_min, cz, characteristic_length)
    p2 = gmsh.model.geo.addPoint(x_max, y_min, cz, characteristic_length)
    p3 = gmsh.model.geo.addPoint(x_max, y_max, cz, characteristic_length)
    p4 = gmsh.model.geo.addPoint(x_min, y_max, cz, characteristic_length)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Delaunay
    gmsh.model.mesh.generate(2)
    
    elements_tri = gmsh.model.mesh.getElements(2)
    nodes_tri = gmsh.model.mesh.getNodes()
    num_tris = len(elements_tri[1][0]) if len(elements_tri[1]) > 0 else 0
    num_nodes_tri = len(nodes_tri[0])
    
    gmsh.write("comparison_triangular.msh")
    gmsh.finalize()
    
    print(f"  Nodes: {num_nodes_tri}")
    print(f"  Triangles: {num_tris}")
    
    # Generate quad mesh
    print("\nGenerating QUAD mesh...")
    num_nodes_quad, num_quads = create_unstructured_quad_mesh(
        "comparison_quad.msh",
        characteristic_length,
        geometry_params,
        recombine_algorithm=1
    )
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Mesh Type':<20} {'Nodes':<15} {'Elements':<15} {'Avg Nodes/Element':<20}")
    print("-"*80)
    print(f"{'Triangular':<20} {num_nodes_tri:<15} {num_tris:<15} {num_nodes_tri/num_tris:.2f}")
    print(f"{'Quadrilateral':<20} {num_nodes_quad:<15} {num_quads:<15} {num_nodes_quad/num_quads:.2f}")
    print("="*80)
    print(f"\nElement count ratio (Tri/Quad): {num_tris/num_quads:.2f}")
    print("Note: Typically need ~2× triangles to match quad coverage\n")


# =============================================================================
# MAIN EXECUTION EXAMPLES
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("GMSH QUADRILATERAL MESH GENERATOR")
    print("="*80 + "\n")
    
    # Example 1: Create structured quad mesh suite
    print("="*80)
    print("EXAMPLE 1: Structured Quad Mesh Suite")
    print("="*80)
    
    wavelength = 1.0
    mesh_info = create_quad_mesh_suite(
        wavelength=wavelength,
        divisions_list=[5, 10, 15, 20, 25]
    )
    
    # Example 2: Create single unstructured quad mesh
    print("\n" + "="*80)
    print("EXAMPLE 2: Unstructured Quad Mesh")
    print("="*80 + "\n")
    
    create_unstructured_quad_mesh(
        output_filename="horn_parab_1.msh",
        characteristic_length=1,
        geometry_params={'width': 162.0, 'height': 162.0, 'center': [0, 0, 0]},
        recombine_algorithm=1  # Try 0, 1, 2, or 3 for different algorithms
    )
    
    # Example 3: Compare triangular vs quad
    print("\n" + "="*80)
    print("EXAMPLE 3: Triangle vs Quad Comparison")
    print("="*80 + "\n")
    
    compare_tri_vs_quad_mesh(characteristic_length=0.1)
    
    # Example 4: Mesh STEP file with quads
    # Uncomment to use:
    # print("\n" + "="*80)
    # print("EXAMPLE 4: Quad Mesh from STEP File")
    # print("="*80 + "\n")
    # 
    # create_quad_mesh_from_step(
    #     step_filename="your_geometry.step",
    #     output_filename="geometry_quad.msh",
    #     characteristic_length=0.05
    # )
    
    print("\n" + "="*80)
    print("✅ Quad mesh generation complete!")
    print("="*80)
    print("\nGenerated meshes:")
    print("  - Structured quads: plane_quad_5x5.msh, plane_quad_10x10.msh, etc.")
    print("  - Unstructured quad: plane_quad_unstructured.msh")
    print("  - Comparison meshes: comparison_triangular.msh, comparison_quad.msh")
    print("\nUse these in your MoM code (note: quad elements use different basis functions!)")
    print("="*80 + "\n")