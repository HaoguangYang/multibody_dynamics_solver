#!/usr/bin/env python

from .spatial_transformations import *
import taichi as ti
import taichi.math as tm

vec6 = ti.types.vector(6, ti.float64)

@ti.dataclass
class constrained_motion_problem_solver:
    block_00: mat3
    block_01: mat3
    block_10: mat3
    block_11: mat3

    @ti.func
    def solve(self, lagrangian: vec6) -> vec6:
        return vec6(
            self.block_00 @ lagrangian[0:3] + self.block_01 @ lagrangian[3:6],
            self.block_10 @ lagrangian[0:3] + self.block_11 @ lagrangian[3:6]
        )

@ti.dataclass
class constrained_motion_problem:
    block_00: mat3
    block_01: mat3
    block_10: mat3
    block_11: mat3

    @ti.func
    def analyze(self) -> constrained_motion_problem_solver:
        block_00_inv = self.block_00.inverse()
        block_00_inv_block_01 = block_00_inv @ self.block_01
        block_11 = (self.block_11 - self.block_10 @ block_00_inv_block_01).inverse()
        block_10 = -self.block_10 @ block_00_inv @ block_11
        return constrained_motion_problem_solver(
            block_00 = block_00_inv - block_00_inv_block_01 @ block_10,
            block_01 = -block_00_inv_block_01 @ block_11,
            block_10 = block_10,
            block_11 = block_11
        )

@ti.dataclass
class inertia:
    # [[moment, block_01], [block_10, mass*identity(3)]]
    mass: ti.float64
    moment: mat3
    block_10: mat3
    # block_01 = block_10.transpose()

    @ti.func
    def transform(self, tf: transform) -> ti.template():
        adj = tf.as_adj()
        mb = self.mass * adj.block_10
        return inertia(
            mass = self.mass,
            moment = (adj.rotation.transpose() @ self.moment + adj.block_10.transpose() @ self.block_10) @ adj.rotation +
                    (adj.rotation.transpose() @ self.block_10.transpose() + mb.transpose()) @ adj.block_10,
            block_10 = adj.rotation.transpose() @ self.block_10 @ adj.rotation + adj.rotation.transpose() @ mb
        )

    @ti.func
    def inverse_mult(self, w: wrench) -> acceleration:
        ret = acceleration()
        m_inv = 1. / self.mass
        if self.block_10.any():
            lie_bracket_p = -m_inv * self.block_10
            r_moment_inv_rt = (self.moment + self.block_10.transpose() @ lie_bracket_p).inverse()
            i_inv_block_10 = lie_bracket_p @ r_moment_inv_rt
            i_inv_block_11 = i_inv_block_10 @ lie_bracket_p.transpose() + m_inv * ti.Matrix.identity(ti.float64, 3)
            ret.angular = r_moment_inv_rt @ w.torque + i_inv_block_10.transpose() @ w.force
            ret.linear = i_inv_block_10 @ w.torque + i_inv_block_11 @ w.force
        else:
            ret.angular = self.moment.inverse() @ w.torque
            ret.linear = m_inv * w.force
        return ret

    @ti.func
    def add_constraint(self, dof_mask: int) -> constrained_motion_problem:
        # initialized with block-transposition, such that setting cols to zero is easier
        ret = constrained_motion_problem(
            block_00 = self.moment,
            block_10 = self.block_10.transpose(),
            block_01 = self.block_10,
            block_11 = self.mass * ti.Matrix.identity(ti.float64, 3)
        )
        for dim in ti.static(range(3)):
            # rotational
            if not (dof_mask & (1 << (dim+3))):
                # dimension is fixed. a is known. w is to be solved
                ret.block_00[dim, :] = 0.
                ret.block_00[dim, dim] = -1.
                ret.block_10[dim, :] = 0.
            # translational
            if not (dof_mask & (1 << dim)):
                # dimension is fixed. a is known. w is to be solved
                ret.block_01[dim, :] = 0.
                ret.block_11[dim, dim] = -1.
        ret.block_00 = ret.block_00.transpose()
        ret.block_10 = ret.block_10.transpose()
        ret.block_01 = ret.block_01.transpose()
        return ret

    @ti.func
    def generate_constrained_lagrangian(self, dof_mask: int, w: wrench, a: acceleration) -> vec6:
        ret = vec6(0.)
        b10_transpose = self.block_10.transpose()
        for dim in ti.static(range(3)):
            # rotational
            if not (dof_mask & (1 << (dim+3))):
                # dimension is fixed. a is known. w is to be solved
                ret[0:3] -= a.angular[dim] * self.moment[dim, :]
                ret[3:6] -= a.angular[dim] * b10_transpose[dim, :]
            else:
                # dimension is free. a is to be solved. w is known
                ret[dim] += w.torque[dim]
            # translational
            if not (dof_mask & (1 << dim)):
                # dimension is fixed. a is known. w is to be solved
                ret[0:3] -= a.linear[dim] * self.block_10[dim, :]
                ret[dim+3] -= a.linear[dim] * self.mass
            else:
                # dimension is free. a is to be solved. w is known
                ret[dim+3] += w.force[dim]
        return ret

@ti.func
def inertia_from_cog(mass: ti.float64, moment: ti.template(), \
                     tf_from_cog: transform = transform(rotation = [0., 0., 0., 1.])) -> inertia:
    adj = tf_from_cog.as_adj()
    mb = mass * adj.block_10
    return inertia(
        mass = mass,
        moment = adj.rotation.transpose() @ moment @ adj.rotation + adj.block_10.transpose() @ mb,
        block_10 = adj.rotation.transpose() @ mb
    )

@ti.func
def kinetic_energy(i: inertia, v: twist) -> ti.float64:
    angular_premul = v.angular @ i.moment + v.linear @ i.block_10
    linear_premul = v.angular @ i.block_10.transpose() + v.linear * i.mass
    return 0.5 * (tm.dot(angular_premul, v.angular) + tm.dot(linear_premul, v.linear))

