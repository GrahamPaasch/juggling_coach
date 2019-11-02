# juggling_coach

A video analysis tool that teaches you juggling by identifying problems and helping you to avoid practicing mistakes.

TO RUN: py -3 juggling_coach.py -v videos\5_ball.py

From When Balls Collide by Jack Kalvan and Arthur Lewbel, page 144:
"Imagine having a computerized juggling coach that watches you juggle. It could:
⦁	Show video replay of what you did.
⦁	Calculate and display information: the number of throws and catches, values of all throwing parameters for both hands, etc.
⦁	Track your progress, improvement over time.
⦁	Show you asymmetries in your pattern and places where collisions are likely to occur, and then give advice to fix the problem (ex. Increase left hand flight distance by 4 inches. Move right hand throw point 2 inches closer to center.)
⦁	Give coaching advice (ex. Try just a flash... Try it again... Try throwing higher... Relax.)"

The intent of this code repo to calculate and display information. 

The Claude Shannon juggling theorem states that:
h(d + f) = n(d + e)
where:
h = number of hands  
d = dwell time (the time a ball spends in a hand between a catch and the next throw)  
f = flight time (the time a ball is in the air between throw and catch)
n = number of balls
e = empty time (the time a hand is empty between a throw and the next catch)

From this, the aforementioned book concludes that these are the most important parameters to track in order to determine the quality of a juggling pattern:
T = d + e
W = f/T
H = The maximum height above the hand that the ball reaches
F = W * ball diameter
D1 = minimum distance between balls on the same arc
P = horizontal distance between two parabolic arcs
Pattern Width = F + P
Emax = maximum throw angle error allowed in a pattern = (D1 - ball diameter)/4*H

And that a perfect 7 ball pattern using 6cm diameter balls has the values:
T = .42s  
W = 2.8
H = 1.7m
F = .65m
D1 = .23m
P = .27m
Pattern Width = .92m
Emax = 1.4 deg

Another bit of data that would be good to track is extraneous movement. How much did the juggler move their feet, wiggle their shoulders, tilt their head, etc. It is generally agreed that better jugglers have less extraneous movement, and have fixed gazes with their eyes.

Also important is the side view. Did the throws arc backward or forward instead of going straight up and down? 

A two camera set up (front view and side view) that can generate a 3-D graph that shows the backward, forward, height and width motion of the balls would provide the most useful data for helping to learn juggling, but even just insights into the asthetics of a pattern from one vantage point is useful.