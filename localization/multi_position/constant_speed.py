'''
Trajectory generator:
- sample an integer scalar SPEED: this will be the constant speed with which the drone will move in the room; it will be in units of delta x/second
    (from which distribution will we sample it?)
- sample a coordinate COORDINATE: this will be the initial coordinate from which the drone will move
- sample an integer scalar STEPS: this will be the number of steps of this trajectory
- sample a (STEPS,2) vector DISPLACEMENTS: this will represent the displacement in each timestamp; this must be sampled from a uniform DISCRETE(!) distribution {-1,1}^2
- compute trajectory TRAJECTORY (STEPS, 2): this will be the position of the drone overtime; 
    its formula is TRAJECTORY[i] = TRAJECTORY[i-1]+SPEED*DISPLACEMENTS[i] (it yields a random walk)

To make it harder we can modify the formula: TRAJECTORY[i] = TRAJECTORY[i-1]+(SPEED+SPEED_NOISE)*DISPLACEMENTS[i]+DISPLACEMENT_NOISE, SPEED_NOISE ~ N() DISPALECEMENT_NOISE
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Master model:
- call single point model on each point
- collect estimate [and displacement between points?]
- fit a linear regressor on estimate
- give estimates
- compute loss
'''