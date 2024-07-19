from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentRaysReward3(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and reward R_3
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        use_memory: bool = True,
        num_segments: int = 20,
        reach_goal_reward_factor: float = 10,
        **kwargs,
    ):
        super().__init__(
            [
                RobotPositionObservation(),
                GoalPositionObservation(),
                SpeedObservation(),
                AngularVelocityObservation(),
                ReferencePathSampleObservation(1, 0, reference_path_sample_offset),
                ReferencePathCornerObservation(corner_samples),
                SectorAndRayObservation(num_segments, use_memory=use_memory),
                ReachGoalReward(reach_goal_reward_factor, default_val=0.),
                GoalDistanceReward(1, strictly_pos=False),
            ],
            generate_map,
            time_step,
            multiply_rwd=False,
            **kwargs,
        )
