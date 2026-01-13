import pyvista as pv

# Load saved mesh
mesh = pv.read("mesh_planeGauss16freq112.vtk")

# Print info (useful for debugging)
print(mesh)
print("Cell data arrays:", mesh.cell_data.keys())
print("Point data arrays:", mesh.point_data.keys())

# Choose scalar field
scalar_name = "Magnitude"  # change if needed

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(
    mesh,
    scalars="Magnitude",
    cmap="jet",          # <-- key change
    clim=[0, 1],         # <-- matches your colorbar (0 to 1)
    show_edges=True,
    edge_color="black",
    smooth_shading=False,
    lighting=False
)

plotter.add_scalar_bar(title="", n_labels=6)
plotter.show_axes()
plotter.show_bounds(grid="front", location="outer", all_edges=True)
plotter.show()
