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

In this work, rewards are deterministic, i.e. they are assigned to states by the real **reward function** $R:S\times A \to \R$, meaning at state $s_t$ the algorithm receives a reward $r_t = R(s_t, a_t)$. It is proven that by adding a strictly **shaping function** $F(s_t, a_t) = \gamma\Phi(s_{t+1}) - \Phi(s_t)$ with state-dependent potentials $\Phi$, the reward function can be modified to

$$
R' = R + F
$$

 without changing the optimal policy: By solving the modified MDP $M' = (S, A, T, \gamma, R')$ we also solve the original MDP $M$.

With a naive goalkinematic potential

$$
\Phi_{naive}(s) =
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
 
achieving states that are not only close, but are also expected to be closer in the next states, will be rewarded higher - their *goaldynamics* are more favorable.
On the other hand, achieving states that are only statically close, or even moving away, will be rewarded lower.
However, such a frequent rewarding enables *reward hacking*, where the agent might oscillate between moving closer and farther from the goal without actually reaching the goal, i.e. the converged policy is not the optimal policy, let alone a successful one. With a **conjunctive goalkinematic potential**

$$
\Phi_{conj}(s) =
\begin{cases}
1, & \text{if } \forall k: d^{(k)} \lt 0 ,\\
-1, & \text{if } \forall k: d^{(k)} \gt 0, \\
0, & \text{otherwise,}
\end{cases}
$$

that type of reward hacking is decreased with higher order $k$ , since isolated goalderivatives are not rewarded anymore. Since they are also not penalized, exploration is increased - this is especially beneficial in the multi-goal environments of the later sections.

(Formal Proofs?)

(Experimental Proof)
(fetchpush)

In $5$ simulations within a custom version of Gymnasium's sparse *FetchPush* environment, where a robotic arm is trained to push an object from a random start position to a random target position, the shaped reward ($k=3, \gamma = 0.99$)
$$
\begin{split}
\^R_1(s,a) &= R(s,a) + F(s,a) \\ 
&= R(s,a) + 0.99 * \Phi_{conj}(s') - \Phi_{conj}(s)
\end{split}
$$
achieved a mean reduction of $...$ in the area under the curve (AUC) of the success rate over epochs, compared to the unshaped baseline reward, which translates to the *sample efficiency improvement* $I_{1}$ by this particular MDP modification. The details of the experiments are described in the full work.

Note: With potential-based $\Phi_{conj}(s)$, this claim requires that the goaldistance(?) and the DGS are part of the observable state space, which is a separate focus in this research. In the experiments, the efficiency gains are substantial enough even with non-observable goaldistance and DGS, i.e. possibly justifying the theoretical violation of the Markov assumption. To also amplify the reward shaping term, the sparse reward limits were changed from the nonpositive 2-tuple $(-1,0)$ to the nonnegative 2-tuple $(0,1)$. These are the only environmental changes to the original *FetchPush*, i.e.

* an *augmentation* of the observation state space,
* a *translation* of the sparse reward limits and
* a *shaping* term to the reward.

The first two changes are kept throughout the sections where this particular environment is used. 


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
> In goal-oriented RL training towards an optimal policy, by reducing the observation space to the goalderivative entries of the DGS vector of a *verbose* goal (e.g. multi-dimensional and contextual), the training can be successful, more efficient and more general.

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