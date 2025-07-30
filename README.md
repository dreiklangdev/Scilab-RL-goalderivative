# (Research) Reward and Observe Higher-Order Goalderivatives (2025)

Nhu Huy Le \
Hamburg University of Technology


This repository researches into possible improvements to Goal-Oriented Reinforcement Learning by making further use of the **Differential Kinematic State (DKS)** based on the distance to the goal - in the following called *goalderivatives*.

The probed improvements include sample-efficiency during training and generality of the resulting policy to unseen goals.

---

(explaining gif)

A **goalderivative** $d^{(k)}$ of order $k>1$ is the rate of change in the scalar distance $d$ (specific to e.g. the $L²$-norm) or its derivatives (velocity $d^{(1)}$, acceleration $d^{(2)}$, jerk $d^{(3)}$ etc.) *towards* a numerically defined goal.

In time-discrete environments, the goalderivatives at a time $t_i$ can be recursively estimated with the distance to the goal (*goaldistance*) at $t_i$ by backward difference after a timestep $\Delta t$:

$$
d^{(0)}(t_i) = d(t_i) := \text{goaldistance at time } t_i
$$

$$
d^{(k)}(t_i) \approx \frac{d^{(k-1)}(t_i) - d^{(k-1)}(t_{i-1})}{t_i - t_{i-1}}
$$

In a vector, multiple goalderivatives of $k > 1$ at $t_i$ form a differential kinematic state vector

$$
s_{DKS}(t_i) = \begin{pmatrix} d^{(1)} \\ d^{(2)} \\ \vdots \\ d^{(k)} \end{pmatrix}(t_i)
$$

The core idea is now to evaluate the vector for either reward engineering, observation augmention, or both. 

> **Summary** 
>
> Instead of possibly using only the distance to the goal, its derivatives are also considered - in the reward function or as part of the observation.


> **Research Claim 1** (Potential-based Shaping)
> 
> In goal-oriented RL training towards an optimal policy, by only adding to the existent rewards a shaping term based on negative changes between DKS vectors, the training can be more efficient without changing the original optimal policy.

(Formal Proof)

(Experimental Proof)
(fetchpush)


> **Research Claim 2** (Reward Design)
> 
> In goal-oriented RL training towards an optimal policy, by designing rewards based on the goalderivative entries of the DKS vector, the training can be successful (i.e. reach and keep its goal) and efficient.

(Formal Proof)

(Experimental Proof)
(fetchpush)


> **Research Claim 3** (Observation Augmentation)
> 
> In goal-oriented RL training towards an optimal policy, by adding the goalderivative entries of the DKS vector to the observation space, the training can be more efficient.

(adding to the markov property)

(Formal Proof?)

(Experimental Proof)
(fetchpush)


> **Research Claim 4** (Observation Reduction)
> 
> In goal-oriented RL training towards an optimal policy, by reducing the observation space to the goalderivative entries of the DKS vector of a *verbose* goal (e.g. multi-dimensional), the training can be successful, efficient and general.

(Experimental Proof)
(fetchpush)


## Case Study: Visual Imitation of Hand Gestures by a Robotic Hand (Multi-Goal RL with Multi-Dimensional Goals)

(reward function granularity)

(hand gesture imitation)


## References
* HER
* HER envs.
* RL bible (barto, sutton)
* MDP
* markov property
* goaldistance
* potential-based shaping
* reward engineering
* observation augmentation
* observation reduction (for generality)
* multi-goal


## TODO
* hyperparams
* overviewing tables
* different algo(s)
* different env(s)
* "proving" graphs
* "proving" gifs