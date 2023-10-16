#!/usr/bin/env python

import taichi as ti

@ti.data_oriented
class TimeIntegrator:
    @staticmethod
    @ti.func
    def state_derivative(state: ti.template(), time: ti.float64) -> ti.template():
        """
        This method is essentially a virtual function. The user inheriting this
        class shall override this method for their specific physical model.
        """
        return ti.zero(state)

    @staticmethod
    @ti.func
    def dt_multiplier(dt: ti.float64, x_dot: ti.template()) -> ti.template():
        return dt * x_dot

    @staticmethod
    @ti.func
    def dx_integrator(x: ti.template(), dx: ti.template()) -> ti.template():
        return x + dx

    @classmethod
    @ti.func
    def integrate_onestep(cls, x: ti.template(), t: ti.float64, dt: ti.float64) -> ti.template():
        """
        This method is essentially a virtual function. The user inheriting this
        class shall override this method for their specific integration method.
        """
        return cls.rk4_onestep(x, t, dt)

    @classmethod
    @ti.func
    def integrate_multistep(cls, x_start: ti.template(), t_start: ti.float64, dt: ti.float64, nsteps: int) -> ti.template():
        x = x_start
        t = t_start
        ti.loop_config(serialize=True)
        for i in range(nsteps):
            x = cls.integrate_onestep(x, t, dt)
            t += dt
        return x

    @classmethod
    @ti.func
    def forward_euler_onestep(cls, x: ti.template(), t: ti.float64, dt: ti.float64) -> ti.template():
        add, mult, d = ti.static(cls.dx_integrator, cls.dt_multiplier, cls.state_derivative)
        return add(x, mult(dt, d(x, t)))

    @classmethod
    @ti.func
    def rk4_onestep(cls, x: ti.template(), t: ti.float64, dt: ti.float64) -> ti.template():
        add, mult, d = ti.static(cls.dx_integrator, cls.dt_multiplier, cls.state_derivative)
        half_dt = dt * 0.5
        k1 = d(x, t)
        t1 = t + half_dt
        k2 = d(add(x, mult(half_dt, k1)), t1)
        k3 = d(add(x, mult(half_dt, k2)), t1)
        t2 = t + dt
        k4 = d(add(x, mult(dt, k3)), t2)
        one_sixth_dt = half_dt / 3.
        one_third_dt = dt / 3.
        return add(add(add(add(x, mult(one_sixth_dt, k1)), mult(one_third_dt, k2)),
                       mult(one_third_dt, k3)), mult(one_sixth_dt, k4))

