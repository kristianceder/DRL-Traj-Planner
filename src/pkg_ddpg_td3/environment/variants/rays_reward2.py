from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentRaysReward2(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and reward R_2
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.1,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        num_segments: int = 8,
        collision_reward_factor: float = 10,
        cross_track_reward_factor: float = 0.1,
        speed_reward_factor: float = 0.5,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
        jerk_factor: float = 0.02,
        angular_jerk_factor: float = 0.02,
    ):
        super().__init__(
            [
                SpeedObservation(),
                AngularVelocityObservation(),
                ReferencePathSampleObservation(1, 0, reference_path_sample_offset),
                ReferencePathCornerObservation(corner_samples),
                SectorAndRayObservation(num_segments, use_memory=True),
                CollisionReward(collision_reward_factor),
                CrossTrackReward(cross_track_reward_factor),
                SpeedReward(speed_reward_factor, reference_speed),
                JerkReward(jerk_factor),
                AngularJerkReward(angular_jerk_factor),
            ],
            generate_map,
            time_step
        )
