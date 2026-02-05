## Differentiable Tube MPC 

This folder contains a implementation of a *differentiable two-layer Tube MPC* controller with a **discrete barrier state (DBaS)** safety embedding and an **IFT/DDP-based gradient computation**.  


### Method overview

- **Uncertain system**:  
  We consider a discrete-time nonlinear system with bounded additive disturbance
  \[
  x_{t+1} = f(x_t, u_t) + w_t,\quad w_t \in \mathbb{W}
  \]
  where \(\mathbb{W}\) is a known compact disturbance set.

- **Two-layer Tube MPC**:
  - **Nominal MPC**:  
    Plans a disturbance-free nominal trajectory \((\bar{x}_0,\dots,\bar{x}_T,\,\bar{u}_0,\dots,\bar{u}_{T-1})\) using the nominal dynamics
    \[
    \bar{x}_{t+1} = f(\bar{x}_t, \bar{u}_t)
    \]
    subject to tightened state and input constraints that account for the tracking error tube.
  - **Auxiliary MPC**:  
    Given the current *real* state \(x_t\) and the nominal plan \((\bar{x},\bar{u})\), a second MPC problem computes a sequence of *auxiliary controls* that drives the real state toward the nominal trajectory while keeping it inside a robust tube around the safe set.

- **Tube concept**:  
  The nominal MPC sees *tightened* constraints that guarantee that if the real state tracks the nominal within a tube, then the real state always satisfies the original constraints despite disturbances \(w_t\).  
  The auxiliary MPC plus the DBaS safety mechanism enforce that the real state remains inside this tube.


### DBaS safety embedding and relaxed barrier

To encode safety, we introduce a **discrete barrier state** \(b_k\) (DBaS). Intuitively:

- \(b_k\) tracks how close the system is to violating state constraints \(h(x) \ge 0\).
- The evolution of \(b_k\) and the constraints linking \(b_k\) and \(x\) enforce that any admissible trajectory remains inside the safe set, or strongly penalizes trajectories that leave it.

Instead of using a hard indicator of safety, the implementation uses a **relaxed inverse barrier** \(B_\alpha\):

- **Inverse barrier**:  
  For a scalar constraint \(h(x) \ge 0\), a classical inverse barrier would scale like \(1/h(x)\), exploding as \(h(x)\to 0^+\).
- **Relaxation** (\(\alpha\)-relaxation):  
  The function is smoothed and regularized to:
  - avoid numerical blow-up exactly at the boundary,
  - remain differentiable everywhere (including slightly infeasible regions),
  - provide a controllable trade-off between strict constraint satisfaction and gradient smoothness.

The relaxed barrier is evaluated along the trajectory and embedded into the overall cost, together with the auxiliary cost terms on \(b_k\). This is implemented in `barrier.py`.


### Differentiable optimal control via IFT + DDP structure

To make the whole Tube MPC pipeline differentiable with respect to parameters (e.g. model parameters, constraint parameters, cost weights), the code does **not** unroll an iterative solver inside an autograd loop.  
Instead, it uses the **Implicit Function Theorem (IFT)** on the KKT conditions of the optimal control problems.

- **Step 1 – KKT system**:  
  For each OCP (nominal, auxiliary), write the first-order necessary conditions (stationarity, dynamics, constraints, multipliers). At the optimum, these form a nonlinear system \(F(z,\theta) = 0\), where:
  - \(z\) is the stacked vector of primal and dual variables (states, controls, multipliers),
  - \(\theta\) are problem parameters.

- **Step 2 – Implicit differentiation**:  
  Under regularity assumptions, IFT gives
  \[
  \frac{\partial z^*}{\partial \theta}
  = -\left(\frac{\partial F}{\partial z}\right)^{-1}\frac{\partial F}{\partial \theta}.
  \]
  Direct inversion is too expensive, so we exploit the special **DDP (Differential Dynamic Programming) structure**:
  - The Hessian and Jacobian blocks are banded and time-structured.
  - A backward sweep followed by a forward sweep solves the required linear systems in \(\mathcal{O}(T)\) rather than a dense \(\mathcal{O}(T^3)\) solve.

- **Step 3 – Efficient backward/forward passes**:  
  `ddp.py` implements this structured backward/forward pass, providing:
  - sensitivities of optimal states and controls w.r.t. \(\theta\),
  - gradient of scalar objectives (e.g. tracking loss, barrier loss) w.r.t. \(\theta\),
  all in a way that is compatible with PyTorch’s autograd.

This yields a **fully differentiable Tube MPC module** that can be embedded inside larger learning systems .


### Online adaptation (Algorithm 2)

The controller supports **online adaptation** of parameters \(\bar{\theta}\) (nominal) and \(\theta\) (auxiliary / safety-related) using IFT-based gradients.

- **Objective**:  
  The adaptation step uses a loss of the form
  \[
  \mathcal{L}(\bar{\theta},\theta)
  = \|x^*(\theta) - \bar{x}(\bar{\theta})\|_2^2
    + \|b^*(\theta)\|_2^2,
  \]
  where:
  - \(\bar{x}(\bar{\theta})\) is the nominal trajectory from the nominal MPC,
  - \(x^*(\theta)\) and \(b^*(\theta)\) are the realized state and barrier trajectories from the auxiliary MPC.

- **Interpretation**:
  - The first term encourages the *real* closed-loop trajectory to align with the *nominal* trajectory.
  - The second term penalizes activation of the barrier state, which corresponds to approaching or violating safety constraints.

- **Update rule**:
  - Use the IFT/DDP machinery to compute \(\nabla_{\bar{\theta},\theta}\mathcal{L}\).
  - Update \(\bar{\theta},\theta\) with a gradient-based optimizer (e.g. SGD/Adam in PyTorch).
  - Repeat online as new data (states, inputs, disturbances) are observed.

The orchestration of nominal MPC, auxiliary MPC, safety embedding, and parameter updates is implemented in `tube_mpc.py`.


### Quick start

1. **Install dependencies** (recommended inside a fresh virtual environment):

```bash
pip install -r diff_tube_mpc_strict_pt/requirements.txt
```

2. **Run an example experiment** (Dubins vehicle):

```bash
python diff_tube_mpc_strict_pt/run_experiment.py \
  --config diff_tube_mpc_strict_pt/configs/dubins.yaml
```

This will:

- construct the Dubins dynamics and safety set,
- build the two-layer Tube MPC controller with DBaS,
- run a closed-loop rollout under disturbances,
- log the trajectories and save plots/results to `diff_tube_mpc_strict_pt/outputs/`.


### Configuration and systems

- **Configs (`configs/`)**:
  - YAML files specify:
    - system type and parameters,
    - horizon length \(T\),
    - cost weights (tracking, control effort, barrier),
    - disturbance bounds \(\mathbb{W}\),
    - DBaS / barrier hyperparameters (e.g. \(\alpha\), penalties),
    - optimization / solver settings.

- **Systems (`core/systems/`)**:
  - Each file defines:
    - dynamics function \(f(x,u)\),
    - state constraint function \(h(x)\) for safety,
    - nominal state and input dimensions,
    - any system-specific utilities (e.g. linearization helpers).

You can add your own system by creating a new file in `core/systems/` and pointing a config file to it.


### Code structure

- `configs/`: YAML configs for each system/experiment.
- `core/`: core algorithm implementation.
  - `barrier.py`: DBaS logic + relaxed inverse barrier \(B_\alpha\) definitions and utilities.
  - `ocp.py`: parametric optimal control problem definitions (Problems 3/5/6, constraints, costs).
  - `ddp.py`: DDP-style backward/forward passes for efficient IFT-based gradients.
  - `tube_mpc.py`: orchestration of nominal + auxiliary MPC, safety embedding, and Algorithm 2 adaptation.
  - `systems/`: dynamics and safety set \(h(x)\) for each benchmark system.
- `run_experiment.py`: main entrypoint for running experiments and saving results.
- `gradient_check.py`: finite-difference sanity checks on gradients for small horizons.



