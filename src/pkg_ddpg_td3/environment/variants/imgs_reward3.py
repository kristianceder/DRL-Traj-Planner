from typing import Optional

from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentImgsReward3(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as image observations
    and reward R_1
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        image_width: int = 54,
        image_height: int = 54,
        image_scale_x: float = 1 / 18,
        image_scale_y: float = 1 / 18,
        image_down_sample: float = 2,
        image_center_x: float = 0.5,
        image_center_y: float = 0.3,
        image_angle: float = 0,
        collision_reward_factor: float = 100,
        reach_goal_reward_factor: float = 100,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
        w1: float = 1/4,
        w2: float = 1/4,
        w3: float = 1/4,
        w4: float = 1/4,
        config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            [
                SpeedObservation(),
                AngularVelocityObservation(),
                ReferencePathSampleObservation(1, 0, reference_path_sample_offset),
                ReferencePathCornerObservation(corner_samples),
                ImageObservation(image_width, image_height, image_scale_x, image_scale_y, image_down_sample, image_center_x, image_center_y, image_angle),
                ReachGoalReward(reach_goal_reward_factor, default_val=0.),
                CollisionReward(collision_reward_factor),
                NormSpeedReward(w1, reference_speed, tau=0.95),
                NormAccelerationReward(w2),
                NormGoalDistanceReward(w3),
                NormCrossTrackReward(w4),
            ],
            generate_map,
            time_step,
            config=config,
            **kwargs,
        )
