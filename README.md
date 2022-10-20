# 194.077-Applied-Deep-Learning
Inspection into the performance and methodical differences of policy-based and valuebased reinforcement learning agents 
____________________________________________________________________________________
## IN A NUTSHELL

Utilize a pygame environment or two (very different ones) to compare how value-based 
vs. policy-based learning algorithms perform. This is with respect to policy
(max. reward) and convergence. I would like to make hypotheses about why 
I believe f.e. a DQN will outperform an A2C on a certain game and the other way 
around, build and train the agents and then analyse whether and why the hypotheses 
holds (ot not)
____________________________________________________________________________________
## REMARKS

I understand this course is not solely focused on the "reinforcement" aspect of learning, however
this is still the most fascinating part to me. This is why I would like to design my project in 
such a way that allows me to dive deeper into this branch of Deep Learning. However, since the 
methodical framework around agent-building is pretty complex on it's own (at least to me) the 
focus will be less on the "deep" part of deep reinforcement learning. What i mean by this is:
My impression is that understanding and correctly applying the algorithmic logic (synergy of multiple networks
(as in actor-critic)), connection to the environment and convergence analysis will consume 
the largest chunk of time, whereas optimising the agents by applying a large number of layers probably
will probably be out of scope.

However, if this proposal is indicating a project not heavy enough on the "deep" learning
side of events, I would alter towards the following: Focus on one REINF-algorithm (either
value- policy-based BUT definitely featuring a neural network), build a basic version and
then try to enhance it by more accomplished and deeper network architectures for a specific 
application
____________________________________________________________________________________
## NECESSARY STEPS

### I) Build up solid understanding about policy-based methods | 

While I have written simple scripts that implement Q-learning and DQ-learning agents
for applications like "Cart-Pole" and "Frozen Lake", I have not come into contact with policy-based methods very intensively.
Sutton& Barto (2nd edition) has given me a theoretical idea about how these methods work
and relate to the value-based ones, however I haven't fully understood the approach, specific methods
and when their application is supposed to be favorable

### II) Establish hypothesis and decide on comparisson parameters |

Based on the theoretical knowledge established in a prior stage, I will draw up hypthosis regarding policy success and 
convergence behavior of an agent in a specific case. It might become necessary to include further benchmark-parameters
in order to draw conclusions from the agent test runs.

### III) Decide on test environment(s) |

Since I want to analyse the differences and areas of applicability of the different model approaches,
rather then explore whether a certain application is realizable at all, I tend towards utilization
of an off-the-shelf sort of environment. Hence, an application that is more or less "standard". While
it would be very cool to emulate what DeepMind has recently done for matrix-multipilcation-optimisation
I am unsure whether it is smart to spend a lot of time on defining an environment at this stage.
My first thought is smth. like pygame for a number of reasons: the evironments satisfy easy integrability,
allow custom definition of rewards (other then openai_gym) and games are a great way to visualize policies and
deduce clues from the agent's behaviour.

### IV) Hacking time I |

Implement the agents and the connection to the environments. 

### V) Hacking time II |

Define reward function.

### VI) Hacking time III |

Make version with deep NNs

### VII) training |

Train agents and watch convergence (find way to measure change in policy
from one episode to later ones)

### IIX) Results |

Sum up results, analysize and draw conclusions

### VII) Application |

Make a comparing demonstration of policies in action for the respective game. If possible highlight actions that
demonstrate significant pecularities of the respective agent 

Note: Uncertain whether II) and III) are seperable, they have some dependencies on each other
_____________________________________________________________________________________
TIME TABLE IN WEEKS (starting with November) ==> SUM: 100h



