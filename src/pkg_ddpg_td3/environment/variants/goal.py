from ..components import *
from .. import MapGenerator
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentGoal(TrajectoryPlannerEnvironment):
    """
    Very simple environment used for preliminary testing
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.1,
        goal_distance_reward_factor: float = 1,
        reach_goal_reward_factor: float = 0,
        time_cost_factor: float = 0.1
    ):
        super().__init__(
            [
                GoalDistanceObservation(),
                GoalAngleObservation(),
                SpeedObservation(),
                AngularVelocityObservation(),
                GoalDistanceReward(goal_distance_reward_factor),
                ReachGoalReward(reach_goal_reward_factor),
                TimeReward(time_cost_factor)
            ],
            generate_map,
            time_step
        )
