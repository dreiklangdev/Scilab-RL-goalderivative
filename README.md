# (Research) Reward and Observe Higher-Order Goalderivatives (2025)

Nhu Huy Le \
Hamburg University of Technology


This repository researches into possible improvements to Goal-Oriented Reinforcement Learning (RL) by evaluating the **Differential Goalkinematic State (DGS)**, whose components are based on the distance to the goal - in the following called goal-directed derivatives or simply *goalderivatives*.

The probed improvements include sample-efficiency during training and generality of the resulting policy to unseen goals.

---

![goalderivs](res/goalderivs.gif)

A **goalderivative** $d^{(k)}$ of order $k>1$ is the rate of change in the scalar distance $d$ (specific to e.g. the $L²$-norm) or its derivatives (velocity $d^{(1)}$, acceleration $d^{(2)}$, jerk $d^{(3)}$ etc.) *towards* a numerically defined goal.

In time-discrete environments, the goalderivatives at a time $t_i$ can be recursively estimated with the distance to the goal (*goaldistance*) at $t_i$ by backward difference:

$$
d^{(0)}(t_i) = d(t_i) := \text{goaldistance at time } t_i
$$

$$
d^{(k)}(t_i) \approx \frac{d^{(k-1)}(t_i) - d^{(k-1)}(t_{i-1})}{t_i - t_{i-1}}
$$

In a vector, multiple goalderivatives of $k > 1$ at $t_i$ form a differential kinematic state vector

$$
s_{DGS}(t_i) = \begin{pmatrix} d^{(1)} \\ d^{(2)} \\ \vdots \\ d^{(k)} \end{pmatrix}(t_i)
$$

The idea is now to evaluate the vector for either reward design, observation augmentation, or both.

> **Summary** 
>
> Instead of using only the distance to the goal, its derivatives are also considered - in the reward function or as part of the observation in RL.

RL algorithms are applied to a formal **Markov Decision Process (MDP)** 

$$
\begin{split}
M &= (\text{states, actions, transition probabilities, discount factor, rewards}) \\
&= (S, A, T, \gamma, R)
\end{split}
$$

to find, for each **state** $s_k \in S$, the optimal **action** $a^*_{k} \in A$ that maximizes the expected $\gamma$ - discounted return 

$$
G = \mathbb{E} [\sum^{\infty}_{t=0} \gamma^t r_t]
$$

of experienced **rewards** $\{r', r'',\dots\} \subseteq R$ from its next states $\{s', s'',\dots\} \subseteq S$, which are reached with probabilities in $T$. However with RL, $T$ and $R$ are initially unknown and must be gradually discovered by exploration similar to *trial and error*.

The result is an optimal **policy** $\pi^* : S \to A$ that assigns to each state its optimal action. Note that a higher-level goal is not formalized in the MDP - its optimality does not ensure the objective success in reaching (or keeping) an indirect goal. Careful and effective design of the underlying MDP is therefore crucial for the correct and efficient convergence of RL.

> **Research Claim 1** (Goalkinematic Reward Shaping)
> 
> In goal-oriented RL training towards an optimal policy, by adding to the existent rewards a shaping term based on the DGS vector, the training can be more efficient without changing the original optimal policy.

In this work, rewards are deterministic, i.e. they are assigned to states by the real **reward function** $R:S\times A \to \R$, meaning at state $s_t$ the algorithm receives a reward $r_t = R(s_t, a_t)$. It is proven that by adding a strictly **shaping function** $F(s_t, a_t) = \gamma\Phi(s') - \Phi(s)$ with state-dependent potentials $\Phi$, the reward function can be modified to

$$
R' = R + F
$$

 without changing the optimal policy: By solving the modified MDP $M' = (S, A, T, \gamma, R')$ we also solve the original MDP $M$.

With a goalkinematic potential

$$
\Phi_{goal}(s) =
-
\left\lVert 
\begin{pmatrix}
d \\ s_{DGS}
\end{pmatrix}
\right\rVert_2
=
-
\left\lVert 
\begin{pmatrix}
d \\ d^{(1)} \\ d^{(2)} \\ \vdots \\ d^{(k-1)}
\end{pmatrix}
\right\rVert_2,
$$
 
achieving states that are not only close, but also are expected to be closer in the next states, is higher rewarded - their *goaldynamics* are more favorable.
On the other hand, achieving states that are only statically close, or even moving away, is rewarded lower.

In $10$ simulations within Gymnasium's *FetchPush* environment, where a 7-DoF robotic arm is trained to push a cube-shaped object from a start position to a target position, the shaped $\^R^1$ achieved a mean reduction of $...$ in training steps compared to the identical, but unshaped environment (baseline), which is the *sample efficiency improvement* $I^{1}$ by this particular MDP modification. The details of the experiments are described in the full work.


(Formal Proof)

(Experimental Proof)
(fetchpush)

Note: With $\Phi_{goal}(s)$, this claim assumes that the goaldistance and the DGS are part of the observable state space, which is also a separate focus in this research. In the experiments, the efficiency gains are substantial enough even with non-observable DGS, i.e. possibly justifying the theoretical violation of the Markov assumption.

Note 2: In finite-horizon environments and with a discount factor close to 1, $\^R^1$ can be approximated by the undiscounted
$
\^R^{1'}
= R -
\left\lVert
s_{DGS}
\right\rVert_2
$
 , which might change the optimal policy, however. In $10$ experiments with $\^R^{1'}$, $I^{1'}$ was $?$ while still being able to maintain the same success conditions of the original environment.

> **Research Claim 2** (Goalkinematic Reward Design)
> 
> In goal-oriented RL training towards an optimal policy, by designing rewards based on the goalderivative entries of the DGS vector, the training can be successful (i.e. the policy reaches and keeps the goal) and more efficient.

(Formal Proof)

(Experimental Proof)
(fetchpush)


> **Research Claim 3** (Goalkinematic Observation Augmentation)
> 
> In goal-oriented RL training towards an optimal policy, by adding the goalderivative entries of the DGS vector to the observation space, the training can be more efficient.

(adding to the markov property)

(Formal Proof?)

(Experimental Proof)
(fetchpush)


> **Research Claim 4** (Goalkinematic Observation Reduction)
> 
> In goal-oriented RL training towards an optimal policy, by reducing the observation space to the goalderivative entries of the DGS vector of a *verbose* goal (e.g. multi-dimensional), the training can be successful, more efficient and more general.

(Experimental Proof)
(fetchpush)


## Case Study: Fluent Visual Imitation of Hand Gestures by a Robotic Hand (Multi-Goal RL with Multi-Dimensional Goals)

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
* "proving" imgs
* "proving" graphs (eg. training process metric?)
* "proving" gifs (eg. sped up training process video?)
* disclaimer (autonomity)
* license