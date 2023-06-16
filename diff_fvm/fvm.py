from phi import math
from phi.math import extrapolation, Extrapolation, Tensor

from phi.geom import UnstructuredMesh


def at_faces(data: Tensor, mesh: UnstructuredMesh, extrapolation: Extrapolation, scheme='upwind-linear', upwind_vectors: Tensor = None, gradient: Tensor = None):
    if '~neighbors' in data.shape:
        return data
    neighbor_val = mesh.pad_boundary(data, extrapolation)
    if scheme == 'upwind-linear':
        flows_out = (upwind_vectors or data).vector @ mesh.face_normals.vector >= 0
        if gradient is None:
            gradient = spatial_gradient(data, mesh, extrapolation)
        neighbor_grad = mesh.pad_boundary(gradient, gradient.extrapolation)
        interpolated_from_self = data + gradient.vector.dual @ (mesh.face_centers - mesh.center).vector
        interpolated_from_neighbor = neighbor_val + neighbor_grad.vector.dual @ (mesh.face_centers - (mesh.center + mesh.neighbor_offsets)).vector
        # ToDo limiter
        return math.where(flows_out, interpolated_from_self, interpolated_from_neighbor)
    elif scheme == 'upwind':
        flows_out = (upwind_vectors or data).vector @ mesh.face_normals.vector >= 0
        return math.where(flows_out, data, neighbor_val)
    elif scheme == 'linear':
        return (1 - mesh.relative_face_distance) * data + mesh.relative_face_distance * neighbor_val
    else:
        raise NotImplementedError(f"Scheme '{scheme}' not supported for resampling mesh values to faces")


def spatial_gradient(data: Tensor,
                     mesh: UnstructuredMesh,
                     extrapolation: Extrapolation,
                     at: str = 'center',
                     scheme: str = 'linear') -> Tensor:
    if scheme == 'linear':  # Green-Gauss gradient   https://youtu.be/E9_kyXjtRHc?t=863  https://youtu.be/oeA1Bg9GqQQ?t=1171
        face_val = at_faces(data, mesh, extrapolation, scheme=scheme)
        return mesh.integrate_surface(math.c2d(mesh.face_normals) * face_val)
        # grad = math.sum(math.c2d(mesh.face_normals * mesh.face_areas) * face_val, mesh.face_shape.dual) / mesh.volume
    raise ValueError(f"Unsupported scheme: {scheme}")


def divergence(data: Tensor,
               mesh: UnstructuredMesh,
               extrapolation: Extrapolation,
               scheme: str = 'linear',
               upwind_vectors: Tensor = None) -> Tensor:
    data = at_faces(data, mesh, extrapolation, scheme=scheme, upwind_vectors=upwind_vectors)
    return mesh.integrate_surface(data.vector @ mesh.face_normals.vector)


def laplace(data: Tensor,
            mesh: UnstructuredMesh,
            prev_grad: Tensor) -> Tensor:
    neighbor_val = mesh.pad_boundary(data, extrapolation)
    connecting_grad = (mesh.connectivity * neighbor_val - data) / mesh.neighbor_distances
    if prev_grad is not None:  # skewness correction
        prev_grad = prev_grad.at_faces()
        n1 = (mesh.face_normals.vector @ mesh.neighbor_offsets.vector) * mesh.neighbor_offsets / mesh.neighbor_distances ** 2
        n2 = mesh.face_normals - n1
        ortho_correction = prev_grad.vector @ n2.vector
        grad = connecting_grad * math.vec_length(n1) + ortho_correction
    else:
        grad = connecting_grad
    return mesh.integrate_surface(grad)


def convection(velocity: Tensor, mesh: UnstructuredMesh, prev_velocity: Tensor):  # https://youtu.be/E9_kyXjtRHc?t=1035
    velocity = velocity.at_faces(scheme='linear-upwind')
    prev_velocity = prev_velocity.at_faces(scheme='linear-upwind')
    return mesh.integrate_surface(velocity * (prev_velocity.vector @ mesh.face_normals.vector))
    # conv = math.sum(velocity * (prev_velocity.vector @ mesh.face_normals.vector) * mesh.face_areas, dual) / mesh.volume

