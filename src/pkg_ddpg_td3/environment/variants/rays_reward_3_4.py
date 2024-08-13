
from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentRaysReward34(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and a reward multiplication following

    $$R = max(d_t - d_{t-1}, 0) * is_collided * r_speed + r_goal * (150 / timesteps) $$

    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        n_speed_observations: int = 1,
        use_memory: bool = True,
        num_segments: int = 40,
        reach_goal_reward_factor: float = 50,
        goal_distance_factor: float = 1.0,
        speed_factor: float = 4.0,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
        acc_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            [
                RobotPositionObservation(),
                GoalPositionObservation(),
                SpeedObservation(max_len=n_speed_observations),
                AngularVelocityObservation(max_len=n_speed_observations),
                ReferencePathSampleObservation(1, 0, reference_path_sample_offset),
                ReferencePathCornerObservation(corner_samples),
                SectorAndRayObservation(num_segments, use_memory=use_memory),
                ReachGoalReward(reach_goal_reward_factor, default_val=0.),
                CollisionReward(10),
                GoalDistanceReward(goal_distance_factor, strictly_pos=False),
                SpeedReward(speed_factor, reference_speed, tau=0.9),
                AccelerationReward(acc_factor, 1),
                AngularAccelerationReward(acc_factor, 1),
            ],
            generate_map,
            time_step,
            reward_mode='curriculum',
            **kwargs,
        )
