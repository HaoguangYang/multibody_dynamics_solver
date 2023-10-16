#!/usr/bin/env python

from multibody_dynamics_solver.spatial_transformations import *
from multibody_dynamics_solver.rigid_body_mechanics import *
from multibody_dynamics_solver.time_integrator import TimeIntegrator
import taichi as ti
import taichi.math as tm

@ti.kernel
def test_transform_chain(t_in: ti.types.ndarray()) -> transform:
    ret = transform(translation = [0.,0.,0.], rotation = [0., 0., 0., 1.])
    ti.loop_config(serialize=True)
    for n in range(t_in.shape[0]):
        trans = ti.math.vec3(t_in[n, 0], t_in[n, 1], t_in[n, 2])
        rot = quat(t_in[n, 3], t_in[n, 4], t_in[n, 5], t_in[n, 6]).normalized()
        t = transform(translation = trans, rotation = rot)
        ret = t.mul(ret).mul(t.inverse())
    return ret

@ti.kernel
def test_transform_chain_as_matrix(t_in: ti.types.ndarray()) -> ti.math.mat4:
    ret = ti.Matrix.identity(ti.float64, 4)
    ti.loop_config(serialize=True)
    for n in range(t_in.shape[0]):
        trans = ti.math.vec3(t_in[n, 0], t_in[n, 1], t_in[n, 2])
        rot = quat(t_in[n, 3], t_in[n, 4], t_in[n, 5], t_in[n, 6]).normalized()
        t = transform(translation = trans, rotation = rot)
        ret = t.as_htm() @ ret @ t.inverse().as_htm()
    return ret

@ti.kernel
def compare_kinetic_energy_in_three_frames(v: twist, mass: ti.float64, moment: ti.math.mat3,
                                         cog_to_vel_ref: transform, vel_ref_to_another: transform) -> ti.math.vec3:
    cog_inertia = inertia(mass = mass, moment = moment)
    vel_frame_inertia = inertia_from_cog(mass, moment, cog_to_vel_ref)
    other_frame_inertia = vel_frame_inertia.transform(vel_ref_to_another)
    ke_in_current_frame = kinetic_energy(vel_frame_inertia, v)
    ke_in_cog = kinetic_energy(cog_inertia, cog_to_vel_ref.transform_twist_body(v))
    ke_in_another_frame = kinetic_energy(other_frame_inertia, vel_ref_to_another.inverse().transform_twist_body(v))
    return ti.math.vec3(ke_in_current_frame, ke_in_cog, ke_in_another_frame)

class RotationIntegratorTest(TimeIntegrator):
    @staticmethod
    @ti.func
    def state_derivative(state: quat, time: ti.float64) -> vec3:
        # this is the angular velocity
        val = tm.pi/tm.sqrt(3) * ti.math.sin(time)
        return vec3(val)

    @staticmethod
    @ti.func
    def dt_multiplier(dt: ti.float64, x_dot: vec3) -> quat:
        return quat_from_delta_rotation(x_dot, dt)

    @staticmethod
    @ti.func
    def dx_integrator(x: quat, dx: quat) -> quat:
        q = quat_mul(x, dx)
        return q/q.norm()

    @classmethod
    @ti.func
    def integrate_onestep(cls, x: quat, t: ti.float64, dt: ti.float64) -> quat:
        return cls.rk4_onestep(x, t, dt)

@ti.kernel
def test_rot_integrate(x_start: quat, t_start: ti.float64, dt: ti.float64, nsteps: int) -> quat:
    return RotationIntegratorTest.integrate_multistep(x_start, t_start, dt, nsteps)

import numpy as np
import time

ti.init()
t = np.random.randn(7000000)
t.resize([1000000,7])
tic = time.time()
ret1 = test_transform_chain(t)
toc = time.time()
print(ret1)
print(f"Time taken for `test_transform_chain` {t.shape[0]} steps: {toc - tic}")
tic = time.time()
ret2 = test_transform_chain_as_matrix(t)
toc = time.time()
print(ret2)
print(f"Time taken for `test_transform_chain_as_matrix` {t.shape[0]} steps: {toc - tic}")

vel = twist(angular = [1., 2., 3.], linear = [4., 5., 6.])
res = compare_kinetic_energy_in_three_frames(vel, 100.0, np.reshape(100.0*np.random.randn(9), [3,3]),
    transform(translation = [-2, -1, -3], rotation = t[0,3:7]/np.linalg.norm(t[0,3:7])),
    transform(translation = [3, -2, 1], rotation = t[1,3:7]/np.linalg.norm(t[1,3:7]))
)
print("Kinetic energy of a same object in three different reference frames:")
print(res)

x0 = quat(0., 0., 0., 1.)
t0 = 0.
dt = 0.001
nstep = 1*31415926+3141
tic = time.time()
res = test_rot_integrate(x0, t0, dt, nstep)
toc = time.time()
print(f"Time taken for `test_rot_integrate` {nstep} steps: {toc - tic}")
print(f"Integration result: x = {res}, t = {t0 + dt*nstep}")

p = pose.field(shape=())
p[None] = pose(position=[1.,2.,3.], orientation=[0.,0.,0.,1.])

t = transform.field(shape=())
t[None] = transform()

@ti.kernel
def modifier():
    t[None] = pose_to_transform(p[None])

modifier()
print(t[None])
