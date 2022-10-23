# 194.077-Applied-Deep-Learning
Inspection of the performance and methodical differences of policy-based and value-based reinforcement learning agents based on the (RaceCarBullet environment?)
____________________________________________________________________________________
## IN A NUTSHELL

Utilize an environment that is suitable to both contionous state/action spaces and discretized ones in order to compare how value-based (which can handle the former)
perform against policy-based learning algorithms (which cannot (at least not by conception)). This is with respect to policy
(max. reward) and convergence. I will dive into theory and recent publications that offer guidance on the advantages to the two approaches,  implement an DQN-agent and an actor-critic-agent and then analyse how the results resonate with the theory.

*Personal aim:*
1) Use the course to broaden understanding of different approaches to reinforcement learning there are
and how they relate to another
2) Gain experience in slightly more complicated environment frameworks and in dealing with continous action / state spaces
____________________________________________________________________________________
## REMARKS REGARDING THE SCOPE

Since the methodical framework around agent-building is pretty complex on it's own (at least to me) the 
focus will be less on the "deep" part of deep reinforcement learning. What i mean by this is:
Understanding and correctly applying the algorithmic logic (synergy of multiple networks
(as in actor-critic)), connection to the environment and convergence analysis) will consume 
the largest chunk of time, whereas really making the agents' brains (=neural networks) "deep" by applying 
a larger number of layers will be treated as a optional bonus, depending on how well I progress.
____________________________________________________________________________________
## NECESSARY STEPS

### 1. Solidify understanding about policy-based methods

<details><summary>Get details</summary>
 
*planned:* 10h
 
While I have some experience with Q-learning and DQ-learning agents
for applications like "Cart-Pole", "Frozen Lake" (openai-gym) and control tasks represented in a "Matlab Simulink" environmet, I have not implemented  policy-based methods.
Sutton& Barto (2nd edition)[^1] has given me a theoretical idea about how these methods work
and relate to the value-based ones, however I haven't understood the method in depth when it's application is favorable. I will compose a little introduction to policy-based methods to document the underlying theory [here](#foundations)

</details>


### 2. Establish hypothesis and decide on comparisson parameters 

<details><summary>Get details</summary>
 
*planned:* 5h

Based on the theoretical knowledge established in a prior stage, I will draw up hypthosis regarding **policy success** and 
**convergence behavior** of an agent in a specific case. It might become necessary to include further benchmark-parameters
in order to draw conclusions from the agent test runs.
 
</details>

### 3. Check suitability and decide on environments  

<details><summary>Get details</summary>

*planned:* 8h

*Chooice: PyBullet implementation of RacecarBullet [^7]*

Since I want to analyse the differences and areas of applicability of the different model approaches,
rather then explore whether a certain application is realizable at all, I tend towards utilization
of an open source environment. Such are:

*DeepMind OpenSpiel and Control Suite (pysics-informed), PyBullet, Open AI Gym, TensorFlow TF-Agents, (Meta AI ReAgent), (pygame)*

The evironments satisfy easy integrability,
allow custom definition of rewards and games are a great way to visualize policies and deduce clues from the agent's behaviour. Some of the frameworks  include analytical tools (for convergence etc.)

Most interesting appears the TF-Agent framework, since it is integrated with the tensorflow library, supports Collab use and holds are OpenAIgym Atari suite for discrete action spaces and the MuJoCo environments for contiuous ones. However, MuJoCo only offers a 30day-free-trail, which might be too short for my purpose. As an alternative I might fall backt to PyBullet, which is opensource.
 
*Note:*
I would have loved to use an exisitng TrackMania Nations [^5] framework in order to train a policy-based agent, however, the contributors suggest 5h training on a modern GPU (something I don't have access to) for acceptable results. I fear running the training on my CPU will lead either to timeouts (since the agent is learning online) or enourmous training sessions.
 
</details>

### 4. Hacking time I 

<details><summary>Get details</summary>
 
*planned:* 15h

Implement the agents and the connection to the environment. 
</details>

### 5. Hacking time II 
 <details><summary>Get details</summary>
 
  *planned:* 12h
  
 Decide on features to use as basis of action-decision (visual input/ simulation "sensor" data/ prepefined state export from reinforcement- evironment?). Experiment and define a final reward function.
</details>
 
### 6. Analysis regarding policy performance and convergence 
<details><summary>Get details</summary>
 
 *planned:* 6h

Train agents and lock convergence.
find a way to measure change in policy from one episode to later ones.
</details>

### 7. (BONUS) Hacking time III 
<details><summary>Get details</summary>
 
 *planned:* 12h
 
 Make version with deep NNs.
</details>

### 8. Results 
<details><summary>Get details</summary>
 
 *planned:* 6h

Sum up results for delivery of Assignment 2.
</details>

### 9. Presentation 
<details><summary>Get details</summary>
 
 *planned:* 4h

Prepare for presentation.
</details>

### 10. Application 
<details><summary>Get details</summary>
 
 *planned:* 6h

Make a comparing demonstration of policies in action for the respective game. If possible highlight actions that
demonstrate significant pecularities of the respective agent 

</details>

*Note:* steps 1) to 3) will run in parallel.
_____________________________________________________________________________________
## Sum of steps: *planned* ~80h
____________________________________________________________________________________
## FOUNDATIONS 
<details><summary>Get details</summary>

### General overview:

1) Model-based (focus on transition function between states, tough to go from model to actual policy)
2) Value-based (learn action-value function, easy to derive policy from a-v-function, but the value function might be very complex, even if policy is super simple ==> sometimes not efficient in learning)
3) Policy-based (we don't learn the value, but simply the policy that obtimizes the value)

1 and 2 don't focus on the real objective: policy

### Policy-Based:
#### Method Derivation:

1) Parametrize policy directly: $pi_{theta}(a|s)=p(a|s,theta$
sigma: NN-weights

2) What do we learn exactly? 
2.1) We define performance $J1(sigma)=v_{pi_{theta}}(s)$ for the starting state 1. We might want this performance measure to be maximized.
2.2) Or the want the weighted average performance of the state-space to be maximized: $J_{average}(theta)= integral_{over_mü}*$v_{pi_{theta}}(s)$ with mü being the partition of time we spent in a certain state when following our policy. ==> a natural approach, since we want to do well in the states that appear often

3) Now that we have set the objective, we need to optimize the policy in a way to fulfill our demand. Gradient accent is useful (since we want to find a maximum): $delta(theta)=gradient_{theta}(J(theta))$

4) How to estimate the gradient? It might sound natural to sample from the policy, obtain a reward R and take the gradient of the (derive R with respect to all $theta_i$. However, that's not possible since R is a numeric value. Instead we use mathematical identity that allows to build the gradient over the expected reward instead of the the expted gradient of the reward (see Sutton&Barto page 325). This is called "score function trick"

5) We want to make 4) useful for sequential rewards. Turns out the Policy Gradient Theorem states that we can just replace the R in the update-formula for with the value function v.

6) We now introduce baselines in order to reduce variance in the update: Let's introduce the baseline function $b(s)$ (which doesn't depend on the action) and hence can be subtracted from the return. We define b to be the Monte Carlo return (=average reward over whole episode). The latter can be estimated by TF (=critic)

#### Actor-Critic[^3]
on policy
Actor: learns policy; updates theta
Critic: learns value; updates w

"Advantage"-A2C: state has a value(=b) and state-action has a value, if we subtract b, the advantage of taking action a remains.

This is usually done simultaneously, but it might be useful to first learn value-function well, before starting to learn to policy.

### Architecture

1) representation: defines what defines the current state $S_t$. Does not only have to be the current observation, but maybe also the prior state (=recurrent network?) $(S_{t-1},O_t)->S_t$

2) value and policy networks (critic and actor) $S -> v$, $S -> pi$

3) n-step TD loss on v.

4) (min 1:16 [^2])we might have to generate a "semi-gradient"=loss from our defined gradient (since tensorflow optimizers demand one). We do this by multiplying the advantage with the likelihood of taking the action taken.

*Notes:* 
* if we let multiple agents explore multiple instances of the same environment and let dem update the shared policy asynchronously training time can be decreased and effects in a single agent can beaveraged out. This is called A3C 
* We need on policy targets (from that exact same step), off policy will introduce bias
* Dataset needs to be GOOD, because a single timestep with bad policy can destroy the process ever after (Trust region policy ==> $pi_{t+1}$ not very very different from $pi_t$)
* Gaussian Policies

### Usefulness

*Downsides*

* Tougher to get off the ground
* Policy does not capture any information about the environment ==> so as soon as environment changes, policy might be useless
* As a result unefficient use of samples (datapoint might not be very useful to the policy, but it might teach a lot about the world) ==> to use this more advanced policy-based-agents also learn value function parallel to policy

*Advantages*

* Policy might turn out to be very simple
* Agent can naturally handle continuous action spaces
* Agent can learn stochastic policies ==> There are simply grid world situations where deterministic policies cannot distinguish seemingly equal states and the agent will end up in a deadlock. Random movement in such an undistinguishable state might be better here. Sencond exmample: Pokergame (we might want to include stochastic actions in order to decrease predictability)
* Agent can learn appropriate levels of exploration (probability for randomness can be different in every state, which isn't possible in value-based policies)

____________________________________________________________________________________

</details>

## RESEARCH, REFERENCES AND LIBRARIES

 [^1:] [Sutton&Barto - Reinforcement Learning](https://inst.eecs.berkeley.edu/~cs188/sp20/assets/files/SuttonBartoIPRLBook2ndEd.pdf)
 
 [^2:] [Deep Mind - Lecture Series](https://www.youtube.com/watch?v=bRfUxQs6xIM)
 
 [^3:] [Asynchronous Methods for Deep Reinforcement Learning](https://paperswithcode.com/paper/asynchronous-methods-for-deep-reinforcement)
 
 [^4:] [Playing Atari with Deep Reinforcement Learning](https://paperswithcode.com/paper/playing-atari-with-deep-reinforcement)
 
 [^5:] [Track Mania Nations Reinforcement Framework](https://github.com/trackmania-rl/tmrl)
 
 [^6:] [Continuous-action Reinforcement Learning for
Playing Racing Games: Comparing SPG to PPO](https://arxiv.org/pdf/2001.05270v1.pdf)
 
 [^7:] [PyBullet docu](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)
