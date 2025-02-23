# Push Object Data Collection

This is my own implementation of [gym-pusht](https://github.com/huggingface/gym-pusht). My goal was to create a similar implementation with the added benefit of being able to generate my own training data, and to potentially use different types of objects as well. If you want to modify the L into something else, you can just edit the polygon coordinates in `object.py`. I also hooked up an N-DOF arm to the data collection sim to make it a little more realistic. The arm cannot self-collide, but it can overlap the pushed object in 2D (I'm imagining it as being above the object in 3D, with the EE extending down to the plane of the object).

`config.py` lets you tune the contact dynamics to your liking.

The next steps in this project involve collecting data, and then training a diffusion policy.


## Data collection example

![Demo](demo.gif)


## Overview

### Forward Kinematics

For each joint $i$ (with link length $L_i$ and joint angle $\theta_i$), the position $(x_{i+1}, y_{i+1})$ is computed as:

$$
\begin{aligned}
x_{i+1} &= x_i + L_i \cos\left(\sum_{j=0}^{i} \theta_j\right), \\
y_{i+1} &= y_i + L_i \sin\left(\sum_{j=0}^{i} \theta_j\right).
\end{aligned}
$$

The end-effector is located at the last joint.

### Inverse Kinematics

The inverse kinematics (IK) problem is solved using SciPy's least-squares optimizer. The cost function $E$ includes two main parts:

1. **End-Effector Position Error:**

$$
E_{\text{pos}} = \|p_{\text{ee}} - p_{\text{target}}\|^2
$$

2. **Collision Penalties:**
   For non-adjacent links, a penalty is added when the distance $d$ between segments is below a threshold $d_{\text{thresh}}$:

$$
E_{\text{collision}} = \begin{cases}
w_{\text{collision}} \cdot \left(d_{\text{thresh}} - d\right) & \text{if } d < d_{\text{thresh}}, \\
0 & \text{otherwise}.
\end{cases}
$$

The overall cost function is a sum of these terms.

### Moment of Inertia for a Polygon

For the T-shaped object, the moment of inertia $I$ is calculated using the formula:

$$
I = \frac{\rho}{12} \sum_{i=0}^{n-1} \left(x_i y_{i+1} - x_{i+1} y_i\right) \left(x_i^2 + x_i x_{i+1} + x_{i+1}^2 + y_i^2 + y_i y_{i+1} + y_{i+1}^2\right),
$$

where the density $\rho$ is computed as:

$$
\rho = \frac{m}{A} \quad \text{with} \quad A = \frac{1}{2} \sum_{i=0}^{n-1} \left(x_i y_{i+1} - x_{i+1} y_i\right).
$$

### Euler Integration for Object Dynamics

The object's linear and angular dynamics are updated using Euler integration.

**Linear Dynamics:**

$$
\begin{aligned}
a &= \frac{F}{m}, \\
v_{\text{new}} &= v_{\text{old}} + a \Delta t, \\
x_{\text{new}} &= x_{\text{old}} + v_{\text{new}} \Delta t.
\end{aligned}
$$

**Angular Dynamics:**

The torque is computed using the 2D cross product:

$$
\tau = r_x F_y - r_y F_x,
$$

and the angular update is:

$$
\begin{aligned}
\alpha &= \frac{\tau}{I}, \\
\omega_{\text{new}} &= \omega_{\text{old}} + \alpha \Delta t, \\
\theta_{\text{new}} &= \theta_{\text{old}} + \omega_{\text{new}} \Delta t.
\end{aligned}
$$

### Contact Force Computation

When the end-effector (with radius $r_{\text{ee}}$) contacts the object, the penetration $\delta$ is:

$$
\delta = r_{\text{ee}} - d,
$$

where $d$ is the distance from the end-effector to the object’s edge. The resulting force is computed as:

$$
F_{\text{contact}} = k_{\text{contact}} \cdot \delta \cdot \hat{n},
$$

and a damping force is added:

$$
F_{\text{damping}} = k_{\text{damping}} \left(v_{\text{ee}} - v_{\text{object}}\right),
$$

yielding the total force:

$$
F_{\text{total}} = F_{\text{contact}} + F_{\text{damping}}.
$$


