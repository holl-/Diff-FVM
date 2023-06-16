from phi import math
from phi.math import extrapolation, Extrapolation, Tensor

from phi.geom import UnstructuredMesh


def at_faces(data: Tensor, mesh: UnstructuredMesh, boundaries: Extrapolation, scheme='upwind-linear', upwind_vectors: Tensor = None, gradient: Tensor = None):
    if '~neighbors' in data.shape:
        return data
    neighbor_val = mesh.pad_boundary(data, boundaries)
    if scheme == 'upwind-linear':
        flows_out = (upwind_vectors or data).vector @ mesh.face_normals.vector >= 0
        if gradient is None:
            gradient = green_gauss_gradient(data, mesh, boundaries)
        neighbor_grad = mesh.pad_boundary(gradient, gradient.boundaries)
        interpolated_from_self = data + gradient.vector.dual @ (mesh.face_centers - mesh.center).vector
        interpolated_from_neighbor = neighbor_val + neighbor_grad.vector.dual @ (mesh.face_centers - (mesh.center + mesh.neighbor_offsets)).vector
        # ToDo limiter
        return math.where(flows_out, interpolated_from_self, interpolated_from_neighbor)
    elif scheme == 'upwind':
        flows_out = (upwind_vectors or data).vector @ mesh.face_normals.vector >= 0
        return math.where(flows_out, data, neighbor_val)
    elif scheme == 'linear':
        nb_center = math.replace_dims(mesh.center, 'cells', math.dual('~neighbors'))
        cell_deltas = math.pairwise_distances(mesh.center, format=mesh.cell_connectivity, default=None)  # x_N - x_P
        face_distance = nb_center - mesh.face_centers[mesh.interior_faces]  # x_N - x_f
        # face_distance = mesh.face_centers[mesh.interior_faces] - mesh.center  # x_f - x_P
        w_interior = (face_distance.vector @ mesh.face_normals.vector) / (cell_deltas.vector @ mesh.face_normals.vector)  # n·(x_N - x_f) / n·(x_N - x_P)
        w = math.concat([w_interior, math.tensor_like(mesh.boundary_connectivity, 0)], '~neighbors')
        return w * data + (1 - w) * neighbor_val
    else:
        raise NotImplementedError(f"Scheme '{scheme}' not supported for resampling mesh values to faces")


def green_gauss_gradient(data: Tensor, mesh: UnstructuredMesh, boundaries: Extrapolation, scheme='linear') -> Tensor:
    face_val = at_faces(data, mesh, boundaries, scheme=scheme)
    return mesh.integrate_surface(math.c2d(mesh.face_normals) * face_val)


def divergence(data: Tensor, mesh: UnstructuredMesh, boundaries: Extrapolation, scheme: str = 'linear', upwind_vectors: Tensor = None) -> Tensor:
    data = at_faces(data, mesh, boundaries, scheme=scheme, upwind_vectors=upwind_vectors)
    return mesh.integrate_surface(data.vector @ mesh.face_normals.vector)


def diffusion(data: Tensor, mesh: UnstructuredMesh, boundaries: Extrapolation, prev_grad: Tensor, dynamic_viscosity=1., scheme='linear') -> Tensor:
    neighbor_val = mesh.pad_boundary(data, boundaries)
    connecting_grad = (mesh.connectivity * neighbor_val - data) / mesh.neighbor_distances  # (T_N - T_P) / d_PN
    if prev_grad is not None:  # skewness correction
        prev_grad = at_faces(prev_grad, mesh, boundaries.spatial_gradient(), scheme=scheme)
        n1 = (mesh.face_normals.vector @ mesh.neighbor_offsets.vector) * mesh.neighbor_offsets / mesh.neighbor_distances ** 2  # (n·d_PN) d_PN / d_PN^2
        n2 = mesh.face_normals - n1
        ortho_correction = prev_grad.vector @ n2.vector
        grad = connecting_grad * math.vec_length(n1) + ortho_correction
    else:
        grad = connecting_grad
    return dynamic_viscosity * mesh.integrate_surface(grad)  # 1/V ∑_f ∇T ν A


def convection(velocity: Tensor, mesh: UnstructuredMesh, boundaries: Extrapolation, prev_velocity: Tensor, density: float = 1, scheme='linear-upwind'):
    """

    Args:
        velocity:
        density:
        mesh:
        boundaries:
        prev_velocity: Velocities from the last time step, ideally sampled at the faces.
        scheme:

    Returns:

    """
    velocity = at_faces(velocity, mesh, boundaries, scheme=scheme)
    prev_velocity = at_faces(prev_velocity, mesh, boundaries, scheme=scheme)
    return density * mesh.integrate_surface(velocity * (prev_velocity.vector @ mesh.face_normals.vector))  # 1/V ∑_f (n·U_prev) U ρ A
