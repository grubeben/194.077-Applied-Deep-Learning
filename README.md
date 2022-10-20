# 194.077-Applied-Deep-Learning
Inspection into the performance and methodical differences of policy-based and value-based reinforcement learning agents
____________________________________________________________________________________
## IN A NUTSHELL

Utilize two (very different) pygame environments to compare how value-based 
vs. policy-based learning algorithms perform. This is with respect to policy
(max. reward) and convergence. I will make hypotheses about why 
I believe f.e. a DQN-agent will outperform an A2C on a certain game and the other way 
around, build and train the agents and then analyse whether and why the hypotheses 
holds (ot not).

*Personal aim:*
Use the course to broaden understanding of different approaches to reinforcement learning there are
and how they relate to another.
____________________________________________________________________________________
## REMARKS REGARDING THE SCOPE

Since the methodical framework around agent-building is pretty complex on it's own (at least to me) the 
focus will be less on the "deep" part of deep reinforcement learning. What i mean by this is:
Understanding and correctly applying the algorithmic logic (synergy of multiple networks
(as in actor-critic)), connection to the environment and convergence analysis) will consume 
the largest chunk of time, whereas really making the agents' brains (=neural networks) "deep" by applying 
a larger number of layers will be treated as a optional bonus, depending on how well I progress.

*Note:*
If this proposal is not focussing enough on the "deep" learning
side of events, please give me a nudge! In that case, I would alter towards the following: 
Focus on one Reinforcement-algorithm (either value- or policy-based BUT definitely featuring a neural network), 
build a basic version (not deep) and then try to enhance it by more accomplished and deeper network architectures for a specific 
application (this would rank in the "beat the stars"-category).
____________________________________________________________________________________
## NECESSARY STEPS

### 1. Solidify understanding about policy-based methods

<details><summary>Get details</summary>
 
*planned:* 10h
 
While I have written simple scripts that implement Q-learning and DQ-learning agents
for applications like "Cart-Pole" and "Frozen Lake" (openai-gym), I have not come into contact with policy-based methods very intensively.
Sutton& Barto (2nd edition) [1] has given me a theoretical idea about how these methods work
and relate to the value-based ones, however I haven't fully understood the approach, specific methods
and when their application is supposed to be favorable.
 
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

Since I want to analyse the differences and areas of applicability of the different model approaches,
rather then explore whether a certain application is realizable at all, I tend towards utilization
of an off-the-shelf sort of environment. Hence, an application that is more or less "standard". While
it would be very cool to emulate what DeepMind has recently done for matrix-multipilcation-optimisation
I am unsure whether it is smart to spend a lot of time on defining an environment at this stage.
My first thought is smth. like pygame for a number of reasons: the evironments satisfy easy integrability,
allow custom definition of rewards (other then openai_gym) and games are a great way to visualize policies and
deduce clues from the agent's behaviour.
 
</details>

### 4. Hacking time I 

<details><summary>Get details</summary>
 
*planned:* 15h

Implement the agents and the connection to the environments. 
</details>

### 5. Hacking time II 
 <details><summary>Get details</summary>
 
  *planned:* 12h
  
  Experiment and define a final reward function.
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

## References
[1] [Sutton&Barto - Reinforcement Learning] (https://inst.eecs.berkeley.edu/~cs188/sp20/assets/files/SuttonBartoIPRLBook2ndEd.pdf)


