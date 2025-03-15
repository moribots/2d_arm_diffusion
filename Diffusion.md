
# Diffusion

Diffusion policy is an innovative method that generates robot actions by starting with pure noise and then iteratively “denoising” that noise until a smooth and coherent action sequence emerges. In this guide, we explain the key concepts and derivations behind diffusion policy in a beginner-friendly way.

---

## 1. The Forward Diffusion Process

### Objective
We begin with a **clean action sequence** $x_0$ and gradually corrupt it by adding noise until the sequence becomes nearly pure noise. This is called the **forward diffusion process**.

### The Diffusion Equation

At each time step $t$, the noisy action sequence $x_t$ is computed as:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
$$

- **$x_0$**: The original (clean) action sequence.
- **$\epsilon$**: Gaussian noise.
- **$\bar{\alpha}_t$**: The cumulative product of coefficients $\alpha_i$ up to time $t$, where each $\alpha_i = 1 - \beta_i$. The schedule $\{\beta_i\}$ determines how much noise is added at each step.

**Intuition:**  
As $t$ increases, $\bar{\alpha}_t$ decreases. This means the contribution of $x_0$ diminishes, and the noise component dominates.

---

## 2. The Reverse Diffusion Process (Denoising)

### Objective
The reverse diffusion process aims to **remove the noise** and recover a clean action sequence. A neural network—usually a 1D U-Net—is trained to predict the noise that was added.

### Step-by-Step Process

1. **Noise Prediction:**

	The network, denoted by $\epsilon_\theta$, takes the noisy sequence $x_t$, the current timestep $t$, and conditioning information (such as the robot’s state and visual cues) as inputs, and predicts the noise:

$$
\epsilon_\theta(x_t, t, \text{cond})
$$

2. **Recovering the Clean Signal:**

	We rearrange the forward equation to compute an estimate of the original signal $x_0$:

$$
x_0^{\text{pred}} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t, \text{cond})}{\sqrt{\bar{\alpha}_t}}
$$

3. **Training Loss:**

	The network is trained using the mean squared error (MSE) loss between the predicted noise and the actual noise:

$$
L = \mathbb{E}\left[\left\| \epsilon - \epsilon_\theta(x_t, t, \text{cond}) \right\|^2\right]
$$

---

## 3. Conditioning the Denoising Process

For effective robot control, the denoising process is **conditioned** on extra information:

- **State Information:**  
  Typically, the robot’s state at two time steps (for example, the end-effector positions at time $t-1$ and $t$) provides clues about the current motion.

- **Visual Information:**  
  Images from the robot’s camera are processed with a convolutional network (e.g., a ResNet18 with the final classification head removed). These features are then passed through a **SpatialSoftmax** layer, which computes keypoints by weighting spatial coordinates.  
  Conceptually, for a flattened feature map $A$ and a coordinate grid $\text{pos_grid}$, the expected coordinate $\mu$ is computed as:

$$
\mu = \sum_i \text{softmax}(A_i)\, \text{pos_grid}_i
$$

- **Time Embeddings:**  
  The diffusion timestep $t$ is encoded using sinusoidal positional embeddings. For a given $t$, the embedding is computed as:

$$
\text{PE}(t) = \Big[\sin\big(t \cdot \omega_1\big), \cos\big(t \cdot \omega_1\big), \dots, \sin\big(t \cdot \omega_{d/2}\big), \cos\big(t \cdot \omega_{d/2}\big)\Big]
$$

These pieces are concatenated to form the **global conditioning vector**. AKA `global_cond` in code:

$$
\text{global_cond} = \text{state} \oplus \text{visual features} \oplus \text{time embedding}
$$

---

## 4. The U-Net Architecture with FiLM Modulation

### U-Net Structure

A U-Net is an encoder-decoder network with **skip connections**:
- **Encoder:** Compresses the noisy input to a lower-dimensional representation.
- **Decoder:** Reconstructs the signal using skip connections to retain fine details.

### FiLM Modulation

**FiLM (Feature-wise Linear Modulation)** injects the conditioning information into each layer of the U-Net. It applies an affine transformation to each feature map:

$$
\text{FiLM}(x, \text{cond}) = x \cdot (1 + \gamma) + \beta
$$

Here, $\gamma$ and $\beta$ are computed from the conditioning vector. This modulation allows the network to adapt its computations based on the state, visual input, and time information.

---

## 5. Accelerated Inference with DDIM

### The Problem with Standard Reverse Diffusion

Reversing the forward process might require hundreds or even 1000 iterations to gradually remove the noise, which is computationally expensive.

### Denoising Diffusion Implicit Models (DDIM)

DDIM is an accelerated, deterministic sampling method that reduces the number of steps required (e.g., to 50 steps).

### DDIM Sampling Steps

1. **Predict the Clean Signal:**

$$
x_0^{\text{pred}} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t, \text{cond})}{\sqrt{\bar{\alpha}_t}}
$$

2. **Deterministic Update:**

	For a selected pair of timesteps $t$ and $t_{\text{next}}$, update:

$$
x_{t_{\text{next}}} = \sqrt{\bar{\alpha}_{t_{\text{next}}}} \, x_0^{\text{pred}} + \sqrt{1 - \bar{\alpha}_{t_{\text{next}}}} \, \epsilon_\theta(x_t, t, \text{cond})
$$

	By setting a parameter $\eta = 0$ (i.e., no extra noise is added), the process becomes deterministic.

**Benefit:**  
Using DDIM significantly speeds up inference because it requires far fewer network forward passes while still generating a high-quality, denoised action sequence.

---

## References

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Deep Spatial Autoencoders for Visuomotor Learning](https://arxiv.org/abs/1509.06113)
- [Diving into Diffusion Policy with LeRobot](https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/)

---

# Simulator

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