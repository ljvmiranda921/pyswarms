============
Introduction
============

It's all a treasure hunt
-------------------------

Imagine that you and your friends are looking for a treasure together. The
treasure is magical, and it rewards not only the one who finds it, but also
those near to it. Your group knows, approximately, where the treasure is, but
not exactly sure of its definite location.

Your group then decided to split up with walkie-talkies and metal detectors.
You use your walkie-talkie to inform everyone of your current position, and
the metal detector to check your proximity to the treasure. In return, you
gain knowledge of your friends' positions, and also their distance from the
treasure.

As a member of the group, you have two options:

* Ignore your friends, and just search for the treasure the way you want it. Problem is, if you didn't find it, and you're far away from it, you get a very low reward.

* Using the information you gather from your group, coordinate and find the treasure together. The best way is to know who is the one nearest to the treasure, and move towards that person.

Here, it is evident that by using the information you can gather from
your friends, you can increase the chances of finding the treasure, and
at the same time maximize the group's reward. This is the basics of
Particle Swarm Optimization (PSO). The group is called the *swarm*,
you are a *particle*, and the treasure is the *global optimum* [CI2007]_.


Particle Swarm Optimization (PSO)
---------------------------------

As with the treasure example, the idea of PSO is to emulate the social
behaviour of birds and fishes by initializing a set of candidate solutions to
search for an optima. Particles are scattered around the search-space, and
they move around it to find the position of the optima. Each particle
represents a candidate solution, and their movements are affected in a
two-fold manner: (1) their cognitive desire to search individually, (2) and
the collective action of the group or its neighbors. It is a fairly simple
concept with profound applications.

One interesting characteristic of PSO is that it does not use the gradient of
the function, thus, objective functions need not to be differentiable.
Moreover, the basic PSO is astonishingly simple. Adding variants to the
original implementation can help it adapt to more complicated problems.

The original PSO algorithm is attributed to Eberhart,
Kennedy, and Shi [IJCNN1995]_ [ICEC2008]_. Nowadays, a lot of variations
in topology, search-space characteristic, constraints, objectives,
are being researched upon to solve a variety of problems.


Why make PySwarms?
------------------

In one of my graduate courses during Masters, my professor asked us to
implement PSO for training a neural network. It was, in all honesty, my
first experience of implementing an algorithm from concept to code. I
found the concept of PSO very endearing, primarily because it gives
us an insight on the advantage of collaboration given a social situation.

When I revisited my course project, I realized that PSO, given enough
variations, can be used to solve a lot of problems: from simple optimization, 
to robotics, and to job-shop scheduling. I then decided to build a
research toolkit that can be extended by the community (us!) and be used
by anyone.

.. rubric:: References

.. [CI2007] A. Engelbrecht, "An Introduction to Computational Intelligence," John Wiley & Sons, 2007.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization," Proceedings of the IEEE International Joint Conference on Neural Networks, 1995, pp. 1942-1948.

.. [ICEC2008] Y. Shi and R.C. Eberhart, "A modified particle swarm optimizer," Proceedings of the IEEE International Conference on Evolutionary Computation, 1998, pp. 69-73.