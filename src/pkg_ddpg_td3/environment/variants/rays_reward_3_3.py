from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentRaysReward33(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and a reward multiplication following

    $$R = max(d_t - d_{t-1}, 0) * is_collided * r_speed + r_goal * (150 / timesteps) $$
    - when is not done, get a term that is only positive when going towards the goal in the right speed and not colliding
    - when done get a reward that is bigger the faster the goal is reached

    Distance part could be replaced by 1 / (1 + dist_to_goal)

    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        use_memory: bool = True,
        num_segments: int = 40,
        reach_goal_reward_factor: float = 10,
        time_penalty: float = 0.02,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
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
                GoalDistanceReward(1., strictly_pos=False),
                # reward for reference speed
                SpeedReward(1.0, reference_speed),
                CollisionReward(10),
                # ExcessiveSpeedPenaltyReward(time_penalty*25., reference_speed=reference_speed),
                # TimeReward(time_penalty/0.2),  # / 0.02 to correct for timestep
            ],
            generate_map,
            time_step,
            multiply_rwd=False,
            **kwargs,
        )
