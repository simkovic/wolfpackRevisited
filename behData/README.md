Behavioral Data
-------------

The data of 41 subjects are included in 41 seperate text files. Each row of a text file presents data from a single trial. 

In blocks 0 and 1 the columns give:
1. subject id
2. block id (0-2)
3. trial order
4. trial id 
5. quadrant layout (see [Evaluation.py](https://github.com/simkovic/wolfpackRevisited/blob/master/code/Evaluation.py#L19> for details)
6. nominal displacement in degrees
7. id of the agents that disappeared
8. orientation of the missing agent when it disappeared in radians
9. x axis of location recalled by subject
10. y axis of location recalled by subject
11. whether the agent was perpendicular (0) or wolf (1)

In block 2 the columns give

7. agent id of the perpendicular agent
8. agent id of the wolf

In block 3 the columns give

4. orientation in radians or degrees (vp > 350).
5. x axis of location recalled by subject
6. y axis of location recalled by subject

Trial and agent ids help to assign a trajectory to each trial and agent. For instance, the position where of the agent 6 appeared in trial 7 is given by the 6th column of the array in [vp300trial007.npy](https://github.com/simkovic/wolfpackRevisited/blob/master/trajData/vp300/vp300trial007.npy).
