# 194.077-Applied-Deep-Learning
Inspection of the performance and methodical differences of policy-based and value-based reinforcement learning agents based (on the RaceCarBullet environment)
____________________________________________________________________________________
## Revision after feedback (1. Nov. 2022)

In order to enable clear focus on every of the steps, I will prioritize implementing an A2C- agent and then and only then work on a DQN-agent and the comparative aspects of the project. Hence, the project outcome shall definitely include:

1) Implementation of a working A2C- agent
2) Employment of the agent in a PyBullet environment

I will treat the following as bonus (whether time will suffice to execute these will become clear along the way):

3) Implementation of a DQ- agent for the same use case
4) Comparison of convergence and policy quality between the two agents

Note that this will affect the order of the tasks below. Those that are not essential to the "must have" goals are marked with (BONUS).
___________________________________________________________________________________
## IN A NUTSHELL

Utilize an environment that is suitable to both continuous state/action spaces and discretized ones in order to compare how value-based (which can handle the former)
perform against policy-based learning algorithms (which cannot (at least not by conception)). This is with respect to policy
(max. reward) and convergence. I will dive into theory and recent publications that offer guidance on the advantages to the two approaches, implement an DQN-agent and an actor-critic-agent and then analyze how the results resonate with the theory.

*Personal goals:*
1) Use the course to broaden understanding of different approaches to reinforcement learning there are
and how they relate to another
2) Gain experience in slightly more complicated environment frameworks and in dealing with continuous action / state spaces
____________________________________________________________________________________
## NECESSARY STEPS

<details><summary>Get details</summary>

### 1. Solidify understanding about policy-based methods

<details><summary>Get details</summary>
 
*planned:* 10h
 
While I have some experience with Q-learning and DQ-learning agents
for applications like "Cart-Pole", "Frozen Lake" (openai-gym) and control tasks represented in a "Matlab Simulink" environmet, I have not implemented  algorithms from the policy-based methods.
Sutton& Barto (2nd edition)[^1] has given me a theoretical idea about how these methods work
and relate to the value-based ones, however I haven't understood the method in depth when its application is favorable. I will compose a little introduction to policy-based methods to document the underlying theory [here](#foundations)

</details>


### 2. (BONUS) Establish hypothesis and decide on comparison parameters 

<details><summary>Get details</summary>
 
*planned:* 5h

Based on the theoretical knowledge established in a prior stage, I will draw up hypothesis regarding **policy success** and 
**convergence behavior** of an agent in a specific case. It might become necessary to include further benchmark-parameters
in order to draw conclusions from the agent test runs.
 
</details>

### 3. Check suitability and decide on environment

<details><summary>Get details</summary>

*planned:* 8h

*Choice: PyBullet implementation of RacecarBullet [^7]*

Since I want to analyze the differences and areas of applicability of the different model approaches,
rather than explore whether a certain application is realizable at all, I tend towards utilization
of an open source environment. Such are:

*DeepMind OpenSpiel and Control Suite (pysics-informed), PyBullet, Open AI Gym, TensorFlow TF-Agents, (Meta AI ReAgent), (pygame)*

The environments satisfy easy integrability,
allow custom definition of rewards and games are a great way to visualize policies and deduce clues from the agent's behavior. Some of the frameworks  include analytical tools (for convergence etc.)

Most interesting appears the TF-Agent framework, since it is integrated with the Tensorflow library, supports Collab use and holds are OpenAIgym Atari suite for discrete action spaces and the MuJoCo environments for contiuous ones. However, MuJoCo only offers a 30day-free-trail, which might be too short for my purpose.
 
*Note:*
I would have loved to use an existing TrackMania Nations [^5] framework in order to train a policy-based agent, however, the contributors suggest 5h training on a modern GPU (something I don't have access to) for acceptable results. I fear running the training on my CPU will lead either to timeouts (since the agent is learning online) or enormous training sessions.
 
</details>

### 4. Hacking time I 

<details><summary>Get details</summary>
 
*planned:* 15h

Implement the A2C- agent and the connection to the environment. 
</details>

### 5. Hacking time II 
 <details><summary>Get details</summary>
 
  *planned:* 12h
  
Decide on features to use as basis of action-decision (visual input/ simulation "sensor" data/ predefined state export from reinforcement- environment?). Experiment and define a final reward function, maybe experiment with auxiliary tasks.
</details>
 

### 7. (BONUS) Hacking time III 
<details><summary>Get details</summary>
 
 *planned:* 12h
 Implement DQ- agent.
 
</details>

### 6. Analysis regarding policy performance and convergence / (BONUS) comparative study
<details><summary>Get details</summary>
 
 *planned:* 10h

Train agent and lock convergence.
find a way to measure change in policy from one episode to later ones.
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
demonstrate significant peculiarities of the respective agent 

</details>
</details>

_____________________________________________________________________________________
## Sum of steps: *planned* ~80h
____________________________________________________________________________________
## FOUNDATIONS 
<details><summary>Get details</summary>

### General overview:

1) Model-based (focus on transition function between states, tough to go from model to actual policy)
2) Value-based (learn action-value function, easy to derive policy from the a-v-function, but the value function might be very complex, even if policy is super simple ==> sometimes not efficient in learning)
3) Policy-based (we don't learn the value, but simply the policy that optimizes the value)

1 and 2 don't focus on the real objective: policy

### Policy-Based:
#### Method Derivation:

1) Parametrize policy directly: $pi_{theta}(a|s)=p(a|s,theta)$
sigma: NN-weights

2) What do we learn exactly? 
2.1) We define performance $J1(sigma)=v_{pi_{theta}}(s)$ for the starting state 1. We might want this performance measure to be maximized.
2.2) Or the want the weighted average performance of the state-space to be maximized: $J_{average}(theta)= integral_{over_{mü}}*v_{pi_{theta}}(s)$ with mü being the partition of time we spent in a certain state when following our policy. ==> a natural approach, since we want to do well in the states that appear often

3) Now that we have set the objective, we need to optimize the policy in a way to fulfill our demand. Gradient accent is useful (since we want to find a maximum): $delta(theta)=gradient_{theta}(J(theta))$

4) How to estimate the gradient? It might sound natural to sample from the policy, obtain a reward R and derive R with respect to all $theta_i$. However, that's not possible since R is a numeric value. Instead we use mathematical identity that allows to build the gradient over the expected reward instead of the the expected gradient of the reward (see Sutton&Barto page 325). This is called "score function trick"
 
5) We want to make 4) useful for sequential rewards and get rid of the sum. Turns out that we can adapt the update-formula in a way that rids of of the quality function and only uses the reward instead. (p.327) The Gradient theorem states now that we can replace the reward with the value function v.

6) We now introduce baselines in order to reduce variance in the update: Let's introduce the baseline function $b(s)=V(s)$ (which doesn't depend on the action).We define b to be the Monte Carlo return (=average reward over whole episode). The advantage is defined as $Q(s,a)-V(s)=R_{t+1}+gamma*V_{s+1}-V_{s}$. The latter can be estimated by TD learning (=critic)


#### Actor-Critic[^3]
on policy
Actor: learns policy; updates theta
Critic: learns value; updates w

"Advantage"-A2C: state has a value(=b) and state-action has a value, if we subtract b, the advantage of taking action a remains.

This is usually done simultaneously, but it might be useful to first learn value-function well, before starting to learn to policy.
 
*Notes:* 
* if we let multiple agents explore multiple instances of the same environment and let dem update the shared policy asynchronously training time can be decreased and effects in a single agent can be averaged out. This is called A3C.
* We need on policy targets (from that exact same step), off policy will introduce bias
* Dataset needs to be GOOD, because a single timestep with bad policy can destroy the process ever after (Trust region policy ==> $pi_{t+1}$ not very very different from $pi_t$)
* Gaussian Policies

### Usefulness

*Downsides*

* Tougher to get off the ground
* Policy does not capture any information about the environment ==> so as soon as environment changes, policy might be useless
* As a result: inefficient use of samples (datapoint might not be very useful to the policy, but it might teach a lot about the world) ==> to use this more advanced policy-based-agents also learn value function parallel to policy

*Advantages*

* Policy might turn out to be very simple
* Agent can naturally handle continuous action spaces
* Agent can learn stochastic policies ==> There are simply grid world situations where deterministic policies cannot distinguish seemingly equal states and the agent will end up in a deadlock. Random movement in such an undistinguishable state might be better here. Second example: Pokergame (we might want to include stochastic actions in order to decrease predictability)
* Agent can learn appropriate levels of exploration (probability for randomness can be different in every state, which isn't possible in value-based policies)
 
____________________________________________________________________________________

</details>

## Architecture and Implementation
<details><summary>Get details</summary>

### Elements we need for an n-step AAC:
 
1) State representation: $S_t$. Does not only have to be the current observation, but maybe also the prior state (=recurrent network?) $(S_{t-1},O_t)->S_t$

2) 2NNs: value- and a policy network (critic(w) and actor(theta)) $S -> v$, $S -> pi$

3) Loss functions (for 1-step AA2C): 
 
   3.1) Critic: We want $TD=R_{t}+gamma*V_{s+1}-V_{s}=A(s,a)$ to be minimal, which is why we define the loss function as $MSE(A)=A(s,a)^2$
 
   3.2) Actor: (min 1:16 [^2]) We have to generate a "semi-gradient"=loss from our defined gradient (since Tensorflow optimizers demand one). We do this by multiplying the advantage with the likelihood of taking the action taken: $A(s,a_t)*log_prob(a_t|s_t)$
   
4) Loss functions (for n-step AAC): 
  4.1) Critic: $R_{t}+R_{t+1}*gamma+ .. +R_{t+n-1}*gamma^{n-1}+gamma^{n}*V_{s+n}*-V_{s}$
  4.2) Actor:$\sum{log_prob(a_t|s_t)}*A(s,a_t)$  for $t=t,..,t+n$

!NOTE!: In order to enable more efficient training and computations we will use only. Only the last network layer will be different in order to faciliate propability or value output. But what does this mean for the loss functions? We simply sum them up!

If we want to penalise large differences between $P(a_{chosen}|s)-P(_i|s), we add a termin for the Entropy-loss

### Algorithm
#### for ending problems (such as the PoleCart, which terminates once the stick is inclined too far to one side)
 1) initialize $s_0$
 2) initialize trace vectors (storage units for store $r_t,..,r_{t+n} and $V_t,..,V_{t+n}$
 3) Loop while $s_t$ is not terminal
 
 3.1) compute action propabilities $probs_a=pi(.|s)$
 
 3.2) choose action $a_t$ by randomly sampling from distribution
 
 3.3) take action $a_t$, observe $r,s_{t+1}
 
 3.4) append $r_t$ $V_t$ to trace vectors
 
 3.5) if len(trace vectors)== batch.size: perform weight update in NN; clear trace vectors
 
 3.6) $s_t$=$s_{t+1}$
 
#### adapting for continuing problems (such as the BulletHopper)
 "for continuing problems without episode boundaries we need
to define performance in terms of the average rate of reward per time step" [^1]
 
 Whhy and what exactly does that mean?
 
 
 
### Classes and files 
 
 1) **agent()** define networks, updates, policy-saving
 
 2) **main()** initiate environemnt, training and visualisation
 
 
___________________________________________________________________________________
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
