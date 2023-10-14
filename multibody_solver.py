#!/usr/bin/env python

from spatial_transformations import *
from time_integrator import TimeIntegrator
from rigid_body_mechanics import *

import taichi as ti
from enum import Enum

class JointType(Enum):
    # dof mask: b'[rotz roty rotx linz liny linx]. 0 -- locked, 1 -- free
    # A revolute joint rotates along rotx, hence has bitmask 32
    # A prismatic joint translates along linx, hence has bitmask 4
    FIXED = 0               # fixed
    PRISMATIC = 1           # x direction translation
    XY_TRANSLATIONAL = 3    # x, y translation
    XYZ_TRANSLATIONAL = 7   # x, y, z translation
    REVOLUTE = 8            # x rotation
    TELESCOPIC = 9          # x translation, x rotation
    # GEAR = 17               # x translation, y rotation. Contact point needs re-calc
    PLANAR = 35             # x, y translation, z rotation
    UNIVERSAL = 48          # y, z rotation
    SHAFT_CONNECTOR = 49    # x translaiton, y, z rotation
    # WHEEL = 51              # x, y translation, y, z rotation. Contact point needs re-calc
    SPHERICAL = 56          # x, y, z rotation
    # BALL_BEARING = 59       # x, y translation, x, y, z rotation. Contact point needs re-calc
    SIX_DOF = 63            # x, y, z translation, x, y, z rotation

gravity = vec3(0., 0., -9.80665)

@ti.dataclass
class State:
    pose: pose
    twist: twist
    acceleration: acceleration
    wrench: wrench

class StateIntegrator(TimeIntegrator):
    @staticmethod
    @ti.func
    def state_derivative(state: State, time: ti.float64) -> State:
        return state

    @staticmethod
    @ti.func
    def dt_multiplier(dt: ti.float64, x_dot: State) -> State:
        delta_twist_angular = x_dot.acceleration.angular * dt
        delta_twist_linear = x_dot.acceleration.linear * dt
        delta_rot = quat_from_delta_rotation(x_dot.twist.angular + delta_twist_angular * 0.5, dt)
        delta_pos = (x_dot.twist.linear + delta_twist_linear * 0.5) * dt
        return State(
            pose = pose(position = delta_pos, orientation = delta_rot),
            twist = twist(angular = delta_twist_angular, linear = delta_twist_linear)
        )

    @staticmethod
    @ti.func
    def dx_integrator(x: State, dx: State) -> State:
        new_pose = pose_to_transform(x.pose).transform_pose(dx.pose)
        new_pose.orientation /= new_pose.orientation.norm()
        return State(
            pose = new_pose,
            twist = twist(
                angular = x.twist.angular + dx.twist.angular,
                linear = x.twist.linear + dx.twist.linear
            ),
            acceleration = x.acceleration,
            wrench = x.wrench
        )

    @classmethod
    @ti.func
    def integrate_onestep(cls, x: State, t: ti.float64, dt: ti.float64) -> State:
        return cls.forward_euler_onestep(x, t, dt)

@ti.data_oriented
class Joint:
    def __init__(self, parent, child, parent_attachment: transform, child_attachment: transform,
                 init_state: State = State(pose = pose(orientation = [0., 0., 0., 1.])),
                 name = "joint", joint_type: JointType = JointType.SIX_DOF):
        self.name: str = name
        self.parent = parent
        self.child = child
        self.dof_mask: int = joint_type.value
        # parent_com -> origin -> ref -> child_com
        self.transforms = transform.field(shape=3)
        self.transforms[0] = parent_attachment      # parent_com -> origin
        self.transforms[1] = child_attachment       # child_com -> ref
        # self.transforms[2] = ref -> child_com
        self.states = State.field(shape=2)     # current state, state setpoint
        self.states[0] = init_state            # initialize
        # TODO: not implemented yet
        self.states[1] = init_state            # initialize
        self.angular_stiffness: vec3 = vec3(0., 0., 0.)
        self.linear_stiffness: vec3 = vec3(0., 0., 0.)
        self.angular_damping: vec3 = vec3(0., 0., 0.)
        self.linear_damping: vec3 = vec3(0., 0., 0.)

    @ti.func
    def setup(self) -> None:
        self.transforms[2] = self.transforms[1].inverse()   # ref -> child_com

    @ti.func
    def update_joint_state(self) -> None:
        """
        update joint pose and apply dof displacement constraints
        Rquires:
        updated self.child.state_com_in_world.pose
        updated self.child.tf_com_to_world = self.child.state_com_in_world.pose.inverse
        """
        # calculate self.state
        self.states[0].pose = self.transforms[2]\
            .mul(pose_to_transform(self.child.state_com_in_world[None].pose).inverse())\
            .mul(pose_to_transform(self.parent.state_com_in_world[None].pose))\
            .mul(self.transforms[0]).inverse().as_pose()
        for dim in ti.static(range(3)):
            # angular
            if not (self.dof_mask & (1 << (dim+3))):
                # dimension is fixed
                self.states[0].pose.orientation[dim] = 0.0
            # linear
            if not (self.dof_mask & (1 << dim)):
                # dimension is fixed
                self.states[0].pose.position[dim] = 0.0
        # normalization
        self.states[0].pose.orientation /= self.states[0].pose.orientation.norm()

    @ti.func
    def update_joint_constraints(self, dt: ti.float64) -> None:
        """
        update joint twist and wrench, and apply dof twist and wrench constraints. Returns the impulse of the constraint.
        Requires:
        updated self.child.tf_com_to_world = self.child.state_com_in_world.pose.inverse
        self.child
        self.child.ext_force_at_com
        dt
        Returns:
        constraint_wrench_at_joint
        """
        # TODO: do this at child link
        tf_world_to_child_com = pose_to_transform(self.child.state_com_in_world[None].pose)
        tf_child_com_to_world = tf_world_to_child_com.inverse()
        # transform world-frame motion state to child com frame, then to joint frame.
        old_twist_at_child_com = tf_child_com_to_world.transform_twist_perspective(self.child.state_com_in_world[None].twist)
        accel_at_child_com = self.child.inertia_com[None].inverse_mult(self.child.state_com_in_world[None].wrench)

        # start of the joint part
        twist_at_ref = self.transforms[2].transform_twist_body(old_twist_at_child_com)
        inertia_at_ref = self.child.inertia_com[None].transform(self.transforms[1])

        tf_ref_to_parent_com = pose_to_transform(self.transforms[0].transform_pose(self.states[0].pose)).inverse()
        parent_twist_at_ref = tf_ref_to_parent_com.transform_twist_body(
            pose_to_transform(self.parent.state_com_in_world[None].pose).inverse().transform_twist_perspective(
                self.parent.state_com_in_world[None].twist
            )
        )

        # constrains twist
        dt_inv = 1. / dt
        accel_constraint_at_ref = acceleration()
        for dim in ti.static(range(3)):
            # angular
            if not (self.dof_mask & (1 << (dim+3))):
                # dimension is fixed
                accel_constraint_at_ref.angular[dim] = (parent_twist_at_ref.angular[dim] - twist_at_ref.angular[dim]) * dt_inv
                # TODO: else: calculate joint twist and accel
            # linear
            if not (self.dof_mask & (1 << dim)):
                # dimension is fixed
                accel_constraint_at_ref.linear[dim] = (parent_twist_at_ref.linear[dim] - twist_at_ref.linear[dim]) * dt_inv
                # TODO: else: calculate joint twist and accel
        l = inertia_at_ref.generate_constrained_lagrangian(self.dof_mask, self.states[1].wrench, accel_constraint_at_ref)
        # TODO: inertia_at_ref.add_constraint().analyze yields a const struct, unless inertia matrix is modified.
        sol = inertia_at_ref.add_constraint(self.dof_mask).analyze().solve(l)
        # dispatch constraint wrench and free accel from sol. this is at ref.
        # TODO: based on twist difference, set wrench controller
        # TODO: based on pose difference, set wrench controller
        for dim in ti.static(range(3)):
            # angular
            if (self.dof_mask & (1 << (dim+3))):
                # dimension is free
                self.states[0].wrench.torque[dim] = self.states[1].wrench.torque[dim]
                accel_constraint_at_ref.angular[dim] = sol[dim]
            else:
                self.states[0].wrench.torque[dim] = sol[dim]
            # linear
            if (self.dof_mask & (1 << dim)):
                # dimension is free
                self.states[0].wrench.force[dim] = self.states[1].wrench.force[dim]
                accel_constraint_at_ref.linear[dim] = sol[dim+3]
            else:
                self.states[0].wrench.force[dim] = sol[dim+3]

        #accel_constraint_at_child_com = self.transforms[1].transform_accel_body(accel_constraint_at_ref, old_twist_at_child_com)
        # rotate the accel half step into the future based on current twist, for improved precesion
        # added 0.01*dt to make the algorithm more stable.
        t_forward = transform(
            rotation = quat_from_delta_rotation(old_twist_at_child_com.angular, dt * 0.501),
        ).mul(self.transforms[1])
        accel_constraint_at_child_com = t_forward.transform_accel_body(accel_constraint_at_ref, old_twist_at_child_com)

        # TODO: return accel_constraint_at_child_com, and update child state in the link method.
        #print("accel@child_com, pre-constraint : ", accel_at_child_com.linear, accel_at_child_com.angular)
        accel_at_child_com.linear += accel_constraint_at_child_com.linear
        accel_at_child_com.angular += accel_constraint_at_child_com.angular
        #print("accel@child_com, post-constraint: ", accel_at_child_com.linear, accel_at_child_com.angular)
        # child setter
        self.child.state_com_in_world[None].acceleration = tf_world_to_child_com.transform_accel_perspective(accel_at_child_com)

@ti.data_oriented
class Link:
    def __init__(self, mass: float, moment: list[float], init_state: State,
                 name = "link"):
        self.name: str = name
        self.mass_py = mass
        self.moment_py: mat3 = mat3(moment)
        self.joints_to_parents : list[Joint] = []
        self.joints_to_children : list[Joint] = []
        # link state -- pose & twist are expressed in external reference (fixed) frame.
        # wrench is expressed in COM frame.
        self.state_com_in_world = State.field(shape=())
        self.state_com_in_world[None] = init_state
        self.inertia_com = inertia.field(shape=())
        # origin frame is for expressing perceptive values
        # TODO: for mounting sensors / porting out visualization. Not implemented.
        self.visual = None
        self.collision = None
        self.origin_to_com: transform = transform(rotation = [0., 0., 0., 1.])

    def assemble(self, another_link, j: Joint) -> None:
        """
        connect this link with another link with a joint
        """
        if j.parent == self and j.child == another_link:
            self.joints_to_children.append(j)
            another_link.joints_to_parents.append(j)
        elif j.child == self and j.parent == another_link:
            self.joints_to_parents.append(j)
            another_link.joints_to_children.append(j)

    @ti.func
    def setup(self) -> None:
        self.inertia_com[None] = inertia_from_cog(self.mass_py, self.moment_py)
        for j in ti.static(range(len(self.joints_to_children))):
            self.joints_to_children[j].setup()
            self.joints_to_children[j].child.setup()

    @ti.func
    def integrate_pose(self, t_start: ti.float64, dt: ti.float64) -> None:
        """
        Periodic update function that integrates motion state of the link.
        Parallel in all links as we are not considering constraints here.
        This needs to be triggered by the kernel to propagate all links.
        """
        # integrate motion in world pose
        # TODO: This integration only applies RK4 to a const acceleration model, which may be not the best.
        # consider doing euler steps, re-evaluating the acceleration, and wrap the whole recursive solver into one RK4 step.
        self.state_com_in_world[None] = StateIntegrator.integrate_onestep(self.state_com_in_world[None], t_start, dt)

    @ti.func
    def apply_ext_forces_at_com(self) -> wrench:
        # this applies only gravity.
        link_gravity = self.inertia_com[None].mass * gravity
        return wrench(
            force = quat_rotate(quat_conj(self.state_com_in_world[None].pose.orientation), link_gravity)
        )

    @ti.func
    def recursive_update(self, dt: ti.float64) -> None:
        """
        Recursive update process. This function is separate from the link or joint classes.
        Update poses from root to leaves:
        Update joint pose state from root to leaves, applying pose constraints. This step zeros out integration errors.
        Re-evaluate link poses using updated joint pose states, from root to leaves.

        Update joint twist from root to leaves, after updating all poses:
        Correct integration errors: apply twist constraints (v -> v@joint -> vc@joint -> vc)
        Update constrained twist (v = vc, v@joint = vc@joint)

        Update acceleration and joint wrenches from leaves to root
        Apply inertial + external force impulse: apply twist constraints ((Fi + F) dt = I@joint dv@joint -> dvc@joint -> (dvc-dv)@joint = Fc@joint dt).
        Recalculate body acceleration (I^-1 (Fi + F + Fc)).
        """
        for j in ti.static(range(len(self.joints_to_parents))):
            self.joints_to_parents[j].update_joint_state()
            self.state_com_in_world[None].pose = pose_to_transform(self.joints_to_parents[j].parent.state_com_in_world[None].pose)\
                .mul(self.joints_to_parents[j].transforms[0])\
                .mul(pose_to_transform(self.joints_to_parents[j].states[0].pose))\
                .mul(self.joints_to_parents[j].transforms[2])\
                .as_pose()

        for j in ti.static(range(len(self.joints_to_children))):
            self.joints_to_children[j].child.recursive_update(dt)

        # this is a static method for each link
        self.state_com_in_world[None].wrench = self.apply_ext_forces_at_com()

        for j in ti.static(range(len(self.joints_to_children))):
            # transform joint frame forces to com frame
            constraint_r_at_com = self.joints_to_children[j].transforms[0]\
                .transform_wrench_body(self.joints_to_children[j].states[0].wrench)
            self.state_com_in_world[None].wrench.force -= constraint_r_at_com.force
            self.state_com_in_world[None].wrench.torque -= constraint_r_at_com.torque

        for j in ti.static(range(len(self.joints_to_parents))):
            self.joints_to_parents[j].update_joint_constraints(dt)

@ti.data_oriented
class StaticLink(Link):
    def __init__(self, init_state: State, name = "static_link"):
        super().__init__(1.e+12, [1.e+12, 0., 0., 0., 1.e+12, 0., 0., 0., 1.e+12], init_state, name)

    @ti.func
    def integrate_pose(self, t_start: ti.float64, dt: ti.float64) -> None:
        pass

    @ti.func
    def apply_ext_forces_at_com(cls) -> wrench:
        return wrench()

    @ti.func
    def recursive_update(self, dt: ti.float64) -> None:
        for j in ti.static(range(len(self.joints_to_parents))):
            self.joints_to_parents[j].update_joint_state()

        for j in ti.static(range(len(self.joints_to_children))):
            self.joints_to_children[j].child.recursive_update(dt)

        # this is a static method for each link
        self.state_com_in_world[None].wrench = wrench()

        for j in ti.static(range(len(self.joints_to_children))):
            # transform joint frame forces to com frame
            constraint_r_at_com = self.joints_to_children[j].transforms[0]\
                .transform_wrench_body(self.joints_to_children[j].states[0].wrench)
            self.state_com_in_world[None].wrench.force -= constraint_r_at_com.force
            self.state_com_in_world[None].wrench.torque -= constraint_r_at_com.torque

        for j in ti.static(range(len(self.joints_to_parents))):
            self.joints_to_parents[j].update_joint_constraints(dt)

@ti.data_oriented
class World:
    def __init__(self, links = list[Link], root_ids = list[int], init_time: ti.float64 = 0.0, dt: ti.float64 = 0.01):
        self.links: list[Link] = links
        self.root_ids: list[int] = root_ids
        self.time = ti.field(ti.float64, shape=())
        self.time[None] = init_time
        self.dt: ti.float64 = dt

    @ti.func
    def setup(self):
        for i in ti.static(range(len(self.root_ids))):
            self.links[self.root_ids[i]].setup()
        for i in ti.static(range(len(self.root_ids))):
            self.links[self.root_ids[i]].recursive_update(self.dt)

    @ti.func
    def update_kinematics_dynamics(self):
        for i in ti.static(range(len(self.links))):
            self.links[i].integrate_pose(self.time[None], self.dt)
        for i in ti.static(range(len(self.root_ids))):
            self.links[self.root_ids[i]].recursive_update(self.dt)
        self.time[None] += self.dt

