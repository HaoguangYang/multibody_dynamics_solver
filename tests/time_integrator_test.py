#!/usr/bin/env python

from multibody_dynamics_solver.time_integrator import TimeIntegrator
import taichi as ti

class TestInt(TimeIntegrator):
    @staticmethod
    @ti.func
    def state_derivative(state, time):
        return ti.math.sin(time) * ti.one(state)

    @classmethod
    @ti.func
    def integrate_onestep(cls, x, t, dt):
        return cls.rk4_onestep(x, t, dt)

@ti.kernel
def test_integrate_with_history(x_all: ti.types.ndarray(), t_start: ti.float64, dt: ti.float64):
    t = t_start
    ti.loop_config(serialize=True)
    for i in range(x_all.shape[0] - 1):
        x_all[i + 1] = TestInt.integrate_onestep(x_all[i], t, dt)
        t += dt

@ti.kernel
def test_integrate(x_start: ti.float64, t_start: ti.float64, dt: ti.float64, nsteps: int) -> ti.float64:
    return TestInt.integrate_multistep(x_start, t_start, dt, nsteps)


import numpy as np
import time

ti.init()
t0 = 0.
dt = 0.001
nstep = 1*31415926
x_hist = np.zeros([nstep + 1,])
print(f"Initial value: x = {x_hist[0]}, t = {t0}")
soln = -np.cos(t0 + dt*nstep) + 1.
tic = time.time()
test_integrate_with_history(x_hist, t0, dt)
toc = time.time()
print(f"Time taken for `test_integrate_with_history` {nstep} steps: {toc - tic}")
print(f"Integration result: x = {x_hist[-1]}, t = {t0 + dt*nstep}, error = {x_hist[-1] - soln}")
x0 = 0.
tic = time.time()
res = test_integrate(x0, t0, dt, nstep)
toc = time.time()
print(f"Time taken for `test_integrate` {nstep} steps: {toc - tic}")
print(f"Integration result: x = {res}, t = {t0 + dt*nstep}, error = {res - soln}")
