#!/usr/bin/env python

import taichi as ti
import taichi.math as tm

vec3 = ti.types.vector(3, ti.float64)
mat3 = ti.types.matrix(3, 3, ti.float64)
mat4 = ti.types.matrix(4, 4, ti.float64)
point = vec3
quat = ti.types.vector(4, ti.float64)

@ti.func
def quat_mul(q1: quat, q2: quat) -> quat:
    """Overrides multiplication operator for the class, to represent q1 * q2 as
    a quaternion multiplication. The quaternion is expressed in the format of:
    q = xi + yj + zk + w

    Args:
        q1 (quat): the first quaternion to be multiplied.
        q2 (quat): the second quaternion to be multiplied.

    Returns:
        quat: the resultant (rotated) quaternion
    """
    return quat(
        q1.w * q2.xyz + q2.w * q1.xyz + tm.cross(q1.xyz, q2.xyz),
        q1.w * q2.w - tm.dot(q1.xyz, q2.xyz)
    )

@ti.func
def quat_conj(q: ti.template()) -> quat:
    """Perform quaternion conjugation q_bar. The quaternion is expressed in the
    format of: q = xi + yj + zk + w

    Args:
        q (quat): the quaternion to be conjugated.

    Returns:
        quat: conjugated quaternion
    """
    return quat(-q.xyz, q.w)

@ti.func
def quat_rotate(q: quat, v: ti.template()) -> vec3:
    """Perform quaternion rotation of vector v. The quaternion is expressed in
    the format of: q = xi + yj + zk + w

    Args:
        q (quat): the quaternion representing the rotation.
        v (taichi.math.vec3): the vector to be rotated.

    Returns:
        taichi.math.vec3: the rotated vector
    """
    norm_quat = tm.normalize(q)
    return vec3(
        quat_mul(quat_mul(norm_quat, quat(v, 0.)), quat_conj(norm_quat)).xyz
    )

@ti.func
def quat_from_angle_axis(angle: ti.float64, axis: vec3) -> quat:
    half_angle = angle * 0.5
    s = tm.sin(half_angle)
    return quat(axis.x * s, axis.y * s, axis.z * s, ti.cos(half_angle))

@ti.func
def quat_from_delta_rotation(angular_velocity: vec3, dt: ti.float64) -> quat:
    ret = quat(0., 0., 0., 1.)
    half_angle_vec = angular_velocity * dt * 0.5
    nsq = (half_angle_vec * half_angle_vec).sum()
    if nsq < 1.0e-12:
        norm_fac = 1./tm.sqrt(1. + nsq)
        ret.xyz = half_angle_vec * norm_fac
        ret.w = norm_fac
    else:
        n = tm.sqrt(nsq)
        ret.xyz = half_angle_vec * tm.sin(n) / n
        ret.w = tm.cos(n)
    return ret

@ti.func
def quat_to_rot_matrix(q: quat) -> mat3:
    qxx = 2. * q.x ** 2
    qxy = 2. * q.x * q.y
    qxz = 2. * q.x * q.z
    qxw = 2. * q.x * q.w
    qyy = 2. * q.y ** 2
    qyz = 2. * q.y * q.z
    qyw = 2. * q.y * q.w
    qzz = 2. * q.z ** 2
    qzw = 2. * q.z * q.w
    qww = 2. * q.w ** 2
    return mat3([
        qww + qxx - 1., qxy - qzw, qxz + qyw,
        qxy + qzw, qww + qyy - 1., qyz - qxw,
        qxz - qyw, qyz + qxw, qww + qzz - 1.
    ])

@ti.dataclass
class pose:
    position: vec3
    orientation: quat

@ti.dataclass
class twist:
    angular: vec3
    linear: vec3

    @ti.func
    def adj(self, x: ti.template()) -> ti.template():
        return twist(
            angular = tm.cross(self.angular, x.angular),
            linear = tm.cross(self.linear, x.angular) + tm.cross(self.angular, x.linear)
        )

@ti.dataclass
class wrench:
    torque: vec3
    force: vec3

@ti.dataclass
class acceleration:
    angular: vec3
    linear: vec3

transform_htm_repr = mat4

@ti.dataclass
class transform_adjoint_repr:
    # [[rotation, 0], [block_10, rotation]]
    rotation: mat3
    block_10: mat3
    # block_01 = zero

    @ti.func
    def mul(self, x: twist) -> twist:
        return twist(
            angular = self.rotation @ x.angular,
            linear = self.block_10 @ x.angular + self.rotation @ x.linear
        )

    @ti.func
    def inverse(self):
        rt = self.rotation.transpose()
        return transform_adjoint_repr(
            rotation = rt,
            block_10 = - rt @ self.block_10 @ rt
        )

@ti.dataclass
class transform:
    translation: vec3
    rotation: quat

    @ti.func
    def inverse(self):
        r_conj = quat_conj(self.rotation)
        return transform(
            translation = -quat_rotate(r_conj, self.translation),
            rotation = r_conj
        )

    @ti.func
    def mul(self, tf: ti.template()) -> ti.template():
        return transform(
            translation = quat_rotate(self.rotation, tf.translation) + self.translation,
            rotation = quat_mul(self.rotation, tf.rotation)
        )

    @ti.func
    def as_htm(self) -> transform_htm_repr:
        r = quat_to_rot_matrix(self.rotation)
        return transform_htm_repr([
            r[0,0], r[0,1], r[0,2], self.translation.x,
            r[1,0], r[1,1], r[1,2], self.translation.y,
            r[2,0], r[2,1], r[2,2], self.translation.z,
            0., 0., 0., 1.
        ])

    @ti.func
    def as_adj(self) -> transform_adjoint_repr:
        r = quat_to_rot_matrix(self.rotation)
        rt = r.transpose()
        return transform_adjoint_repr(
            rotation = r,
            block_10 = ti.Matrix.cols([tm.cross(self.translation, ti.math.vec3(rt[0,:])),
                                        tm.cross(self.translation, ti.math.vec3(rt[1,:])),
                                        tm.cross(self.translation, ti.math.vec3(rt[2,:]))])
        )

    @ti.func
    def as_pose(self) -> pose:
        return pose(
            position = self.translation,
            orientation = self.rotation
        )

    @ti.func
    def transform_pose(self, p: pose) -> pose:
        return pose(
            position = quat_rotate(self.rotation, p.position) + self.translation,
            orientation = quat_mul(self.rotation, p.orientation)
        )

    @ti.func
    def transform_twist_body(self, t: twist) -> twist:
        rot_angular_vel = quat_rotate(self.rotation, t.angular)
        rot_linear_vel = quat_rotate(self.rotation, t.linear)
        return twist(
            angular = rot_angular_vel,
            linear = rot_linear_vel + tm.cross(self.translation, rot_angular_vel)
        )

    @ti.func
    def transform_twist_perspective(self, t: twist) -> twist:
        return twist(
            angular = quat_rotate(self.rotation, t.angular),
            linear = quat_rotate(self.rotation, t.linear)
        )

    @ti.func
    def transform_accel_body(self, a: acceleration, parent_frame_twist: twist) -> acceleration:
        rot_angular_acc = quat_rotate(self.rotation, a.angular)
        rot_linear_acc = quat_rotate(self.rotation, a.linear)
        return acceleration(
            angular = rot_angular_acc,
            linear = rot_linear_acc + tm.cross(self.translation, rot_angular_acc) +
                tm.cross(tm.cross(parent_frame_twist.angular, self.translation), parent_frame_twist.angular)
        )

    @ti.func
    def transform_accel_perspective(self, a: acceleration) -> acceleration:
        return acceleration(
            angular = quat_rotate(self.rotation, a.angular),
            linear = quat_rotate(self.rotation, a.linear)
        )

    @ti.func
    def transform_wrench_body(self, r: wrench) -> wrench:
        rot_torque = quat_rotate(self.rotation, r.torque)
        rot_force = quat_rotate(self.rotation, r.force)
        return wrench(
            torque = rot_torque + tm.cross(self.translation, rot_force),
            force = rot_force
        )

    @ti.func
    def transform_wrench_perspective(self, r: wrench) -> wrench:
        return wrench(
            torque = quat_rotate(self.rotation, r.torque),
            force = quat_rotate(self.rotation, r.force)
        )

@ti.func
def pose_to_transform(p: pose) -> transform:
    return transform(translation = p.position, rotation = p.orientation)
