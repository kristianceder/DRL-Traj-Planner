from typing import Optional
from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentRaysReward33(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and a reward.
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        use_memory: bool = True,
        n_speed_observations: int = 1,
        num_segments: int = 40,
        reach_goal_reward_factor: float = 100,
        collision_factor: float = 100,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
        w1: float = 1/4,
        w2: float = 1/4,
        w3: float = 1/4,
        w4: float = 1/4,
        config: Optional[dict] = None,
        reward_mode: Optional[str] = None,
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
                CollisionReward(collision_factor),
                NormSpeedReward(w1, reference_speed, tau=0.95),
                NormAccelerationReward(w2),
                NormGoalDistanceReward(w3),
                NormCrossTrackReward(w4),
            ],
            generate_map,
            time_step,
            config=config,
            reward_mode=reward_mode,  # TODO remove
            **kwargs,
        )
