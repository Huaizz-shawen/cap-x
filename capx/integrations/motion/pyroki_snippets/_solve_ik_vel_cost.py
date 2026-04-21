"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax import Array
from jaxls import Cost, Var, VarValues


@Cost.create_factory
def limit_velocity_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[Array],
    prev_cfg: Array,
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint velocity limit violations."""
    joint_vel = (vals[joint_var] - prev_cfg) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    prev_cfg: onp.ndarray,
    initial_cfg: onp.ndarray | None = None,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    num_joints = robot.joints.num_actuated_joints

    def _coerce_cfg_shape(cfg: onp.ndarray | None, name: str) -> onp.ndarray | None:
        if cfg is None:
            return None
        arr = onp.asarray(cfg, dtype=onp.float64).reshape(-1)
        if arr.shape[0] == num_joints:
            return arr
        # Some callers pass an augmented robot state (e.g. 7 joints + gripper),
        # while IK only expects actuated arm joints.
        if arr.shape[0] > num_joints:
            return arr[:num_joints]
        # Keep solver stable even when a shorter vector is provided.
        if arr.shape[0] == 0:
            return onp.zeros((num_joints,), dtype=onp.float64)
        pad = onp.repeat(arr[-1], num_joints - arr.shape[0])
        return onp.concatenate([arr, pad], axis=0)

    prev_cfg = _coerce_cfg_shape(prev_cfg, "prev_cfg")
    initial_cfg = _coerce_cfg_shape(initial_cfg, "initial_cfg")

    target_link_index = robot.links.names.index(target_link_name)
    init = None if initial_cfg is None else jnp.array(initial_cfg)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        jnp.array(prev_cfg),
        init,
    )
    assert cfg.shape == (num_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    prev_cfg: jax.Array,
    initial_cfg: jax.Array | None,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
        limit_velocity_cost(
            robot,
            joint_var,
            prev_cfg,
            0.2,  # dt
            2.0,
        ),
    ]
    initial_vals = None
    if initial_cfg is not None:
        initial_vals = jaxls.VarValues.make((joint_var.with_value(initial_cfg),))

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=initial_vals,
        )
    )
    return sol[joint_var]
