from math import cos, sin, pi
import numpy as np
import random
from shapely.geometry import Polygon, JOIN_STYLE

from . import MobileRobot

from typing import Union, Callable, List
from numpy.typing import NDArray, ArrayLike


def _exterior_nodes(polygon: Polygon, orient: int):
    """
    Returns the coordinates of the exterior of a polygon

    :param polygon: The polygon
    :param orient: The orientation of the returned coordinates, 1 for
                   clockwise, -1 for counter clockwise
    """
    exterior = polygon.exterior
    if exterior.is_ccw == (orient > 0):
        coords = exterior.coords[-2::-1]
    else:
        coords = exterior.coords[:-1]
    return np.asarray(coords, dtype=np.float32)

def _orient(nodes: ArrayLike, orient: int):
    """
    Orients the nodes of a polygon

    :param nodes: The vertices of the polygon
    :param orient: The desired polygon orientation, 1 for clockwise, -1 for
                   counter clockwise
    """
    return _exterior_nodes(Polygon(nodes), orient)


class KeyFrame:
    """
    An animation keyframe, describing the position and rotation of an object in
    a specific time instance
    """
    def __init__(self, position: ArrayLike, rotation: float):
        self.position = np.asarray(position, dtype=np.float32)
        self.rotation = rotation
    
    def get_rotation_matrix(self) -> NDArray[np.float32]:
        c = cos(self.rotation)
        s = sin(self.rotation)
        return np.array([[c, -s], [s, c]], dtype=np.float32)


class Animation:
    """
    Class the describing the movement of an obstacle as a cyclic keyframe
    animation
    """
    def __init__(self, time_steps: List[float], keyframes: List[KeyFrame], interp: Callable[[float], float] = lambda x: x, offset: float = 0):
        """
        :param time steps: List of keyframe times, the value at index zero must be 0
        :param keyframes: Keyframes of the animation
        :param interp: The interpolation method
        :param offset: The animation time offset
        """

        assert time_steps[0] == 0, "First keyframe must be valid at t = 0"
        assert len(time_steps) == len(keyframes) + 1, "Time steps must be one more than the number of keyframes (the last time step is for looping)"

        self.time_steps = time_steps
        self.keyframes = keyframes
        self.interp = interp
        self.offset = offset
        self.length = sum(time_steps)
    
    def get_keyframe(self, time: float) -> KeyFrame:
        """
        Returns the keyframe representing the position and angle of the object
        at the desired time
        """
        time = (time + self.offset) % self.length

        t = 0
        for i in range(len(self.keyframes)):
            t += self.time_steps[i]
            if t <= time < t + self.time_steps[i + 1]:
                alpha = self.interp((time - t) / self.time_steps[i + 1])
                k0 = self.keyframes[i]
                k1 = self.keyframes[(i + 1) % len(self.keyframes)]

                return KeyFrame(k0.position * (1 - alpha) + k1.position * alpha, k0.rotation * (1 - alpha) + k1.rotation * alpha)

    @staticmethod
    def static(position: ArrayLike = (0, 0), angle: float = 0):
        """
        An animation with no movement
        """
        return Animation([0, 1], [KeyFrame(position, angle)])
    
    @staticmethod
    def periodic(p1: ArrayLike, p2: ArrayLike, freq: float, angle: float = 0, offset: float = 0):
        """
        A dynamic animation as defined by the MPC paper https://doi.org/10.1109/CASE49439.2021.9551644
        """
        interp = lambda x: (1 - cos(x * pi) ) / 2
        time_step = pi/freq if freq != 0 else 1
        return Animation([0, time_step, time_step], [KeyFrame(p1, angle), KeyFrame(p2, angle)], interp, offset)
    

class Obstacle:
    """
    Class representing an obstacle (static or dynamic) as a polygon with a
    cyclic keyframe animation
    """
    _nodes: NDArray[np.float32]
    _padded_nodes: Union[NDArray[np.float32], None] = None
    _keyframe: KeyFrame
    _padded_polygon: Union[Polygon, None] = None

    def __init__(
        self,
        nodes: ArrayLike,
        visible_on_reference_path: bool, 
        animation: Animation,
        is_static: bool = True
    ):
        """
        :param nodes: Node coordinates of the polygon of this obstacle
        :param visible_on_reference_path: Whether this obstacle should be taken
                                          into account when generating a
                                          reference path
        :param animation: The movement of this obstacle
        """
        self.nodes = _orient(nodes, 1)
        self.visible_on_reference_path = visible_on_reference_path
        self.animation = animation
        self.time = 0
        self.keyframe = animation.get_keyframe(self.time)
        self.is_static = is_static

    def step(self, time_step: float) -> None:
        self.time += time_step
        self.keyframe = self.animation.get_keyframe(self.time)
        
    @property
    def nodes(self) -> NDArray[np.float32]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: ArrayLike) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float32)
        self._padded_polygon = None
        self._padded_nodes = None

    @property
    def padded_nodes(self) -> NDArray[np.float32]:
        # padded_nodes is a cached property
        if self._padded_nodes is None:
            polygon = Polygon(self.nodes)
            self._padded_nodes = _exterior_nodes(polygon.buffer(MobileRobot().cfg.RADIUS, join_style=JOIN_STYLE.round, resolution=4), 1)
        return self._padded_nodes

    @property
    def keyframe(self) -> KeyFrame:
        return self._keyframe

    @keyframe.setter
    def keyframe(self, keyframe: KeyFrame) -> None:
        self._keyframe = keyframe
        self._padded_polygon = None

    @property
    def padded_polygon(self) -> Polygon:
        # padded_polygon is a cached property
        if self._padded_polygon is None:
            self._padded_polygon = Polygon(self.get_padded_vertices())
        return self._padded_polygon

    def get_mitred_vertices(self, inflation_margin:float=None) -> NDArray[np.float32]:
        """
        Returns the coordinates of the corners of the shape of this obstacle,
        padded by the robot radius (mitred padding)
        """
        polygon = Polygon(self.get_vertices())
        if inflation_margin is None:
            inflation_margin = MobileRobot().cfg.RADIUS
        return _exterior_nodes(polygon.buffer(inflation_margin, join_style=JOIN_STYLE.mitre, mitre_limit=2), 1)

    def get_vertices(self) -> NDArray[np.float32]:
        """
        Returns the coordinates of the corners of this obstacle at the current
        simulation time (i.e. including transformations resulting from
        movement)
        """
        return self.keyframe.position + (self.keyframe.get_rotation_matrix() @ self.nodes.T).T

    def get_padded_vertices(self) -> NDArray[np.float32]:
        """
        Returns the coordinates of the corners of this obstacle padded by the
        robot radius at the current simulation time (i.e. including
        transformations resulting from movement) (round padding)
        """
        return self.keyframe.position + (self.keyframe.get_rotation_matrix() @ self.padded_nodes.T).T

    def collides(self, robot: MobileRobot) -> bool:
        """Whether ``robot`` collides with this obstacle"""
        return self.padded_polygon.contains(robot.point)

    @staticmethod
    def create_mpc_static(nodes: ArrayLike, is_static=True) -> 'Obstacle':
        """Creates a static obstacle according to the MPC paper https://doi.org/10.1109/CASE49439.2021.9551644"""
        return Obstacle(nodes, True, Animation.static(), is_static=is_static)


    @staticmethod
    def create_mpc_dynamic(p1: ArrayLike, p2: ArrayLike, freq: float, rx: float, 
                           ry: float, angle: float, corners: int = 12, random: bool = True) -> 'Obstacle':
        """Creates a dynamic obstacle according to the MPC paper https://doi.org/10.1109/CASE49439.2021.9551644
        but added functionality of creating a non-convex object. variable 'random' declares if it should choose
        between convex and non-convex with probability 0.5. If set to false: only convex."""
        if random:
            if np.random.choice([True, False]):
                r = 2
                u_shape = np.array([[-1.4, -0.6], [-1.4, 0.6], [-0.6, 0.6],[-0.6, 0],[0.6, 0], [0.6, 0.6], [1.4, 0.6], [1.4, -0.6],[0.6, -0.6],[0.6, -0.8],[-0.6, -0.8],[-0.6, -0.6]])
                l_shape = np.array([[-1, -0.4], [-1, 0.8], [-0.2, 0.8],[-0.2, 0.4],[1, 0.4], [1, -0.4]])
                # u_shape = np.array([[-r, -r], [-r, r], [r, r],[r, -r],[r/2, -r], [r/2, r/2], [-r/2, r/2], [-r/2, -r]])
                # l_shape = np.array([[0, 0], [0, r], [r/3, r],[r/3, r/3],[r, r/3], [r, 0]])
                scaling_factor = 1
                nodes = np.random.choice(np.array([u_shape, l_shape], dtype=object))*scaling_factor
                offset = 0.5*pi/freq if freq > 0 else 0
                return Obstacle(nodes, False, Animation.periodic(p1, p1, freq, angle, offset=offset))

            else:
                nodes = np.zeros((corners, 2))
                for i in range(corners):
                    angle2 = 2 * pi * i / corners
                    nodes[i, :] = (rx * cos(angle2), -ry * sin(angle2))
                offset = 0.5*pi/freq if freq > 0 else 0
                return Obstacle(nodes, False, Animation.periodic(p1, p2, freq, angle, offset=offset))
        else:
            nodes = np.zeros((corners, 2))
            for i in range(corners):
                angle2 = 2 * pi * i / corners
                nodes[i, :] = (rx * cos(angle2), -ry * sin(angle2))
            offset = np.random.uniform(0,2)*pi/freq if freq > 0 else 0
        return Obstacle(nodes, False, Animation.periodic(p1, p2, freq, angle, offset=offset))

    @staticmethod
    def create_mpc_dynamic_old(p1: ArrayLike, p2: ArrayLike, freq: float, rx: float, ry: float, angle: float, corners: int = 12, is_static=False) -> 'Obstacle':
        """Creates a dynamic obstacle according to the MPC paper https://doi.org/10.1109/CASE49439.2021.9551644"""
        nodes = np.zeros((corners, 2))
        for i in range(corners):
            angle = 2 * pi * i / corners
            nodes[i, :] = (rx * cos(angle), -ry * sin(angle))
        offset = 0 # 0.5*pi/freq if freq > 0 else 0
        return Obstacle(nodes, False, Animation.periodic(p1, p2, freq, angle, offset=offset), is_static=is_static)
    
    @staticmethod
    def create_non_convex_u_shape(p1: ArrayLike, p2: ArrayLike, freq: float, angle: float) -> 'Obstacle':
        """Creates a dynamic obstacle as a non-convex object."""
        r = 1.5
        left = r + random.uniform(-1,1)/2
        right = r + random.uniform(-1,1)/2
        thickness = random.uniform(r/2,r/1.4)
        u_shape = np.array([[-r, -right], [-r, r], [r, r],[r, -left],[thickness, -left], [thickness, thickness], [-thickness, thickness], [-thickness, -right]])
        scaling_factor = 1 + random.uniform(0,1)
        nodes = u_shape*scaling_factor
        offset = 0.5*pi/freq if freq > 0 else 0
        return Obstacle(nodes, False, Animation.periodic(p1, p2, freq, angle, offset=offset))

    @staticmethod
    def create_non_convex_l_shape(p1: ArrayLike, p2: ArrayLike, freq: float, angle: float) -> 'Obstacle':
        """Creates a dynamic obstacle as a non-convex object."""
        r = 1.5
        left = r + random.uniform(0,1)
        right = r + random.uniform(0,1)
        thickness = random.uniform(r/3,r/6)
        l_shape = np.array([[0, 0], [0, left], [thickness, left],[thickness, thickness],[right, thickness], [right, 0]])
        scaling_factor = 1 + random.uniform(0,1)
        nodes = l_shape*scaling_factor
        offset = 0.5*pi/freq if freq > 0 else 0
        return Obstacle(nodes, False, Animation.periodic(p1, p2, freq, angle, offset=offset))


class Boundary:
    """Outer boundary of a map"""
    _vertices: NDArray[np.float32]
    _padded_polygon: Union[Polygon, None] = None

    def __init__(self, vertices: ArrayLike):
        self.vertices = _orient(vertices, -1)

    @property
    def vertices(self) -> NDArray[np.float32]:
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices: ArrayLike) -> None:
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._padded_polygon = None
    
    @property
    def padded_polygon(self) -> Polygon:
        # padded_nodes is a cached property
        if self._padded_polygon is None:
            polygon = Polygon(self.vertices)
            self._padded_polygon = polygon.buffer(-MobileRobot().cfg.RADIUS, join_style=JOIN_STYLE.round, resolution=4)
        return self._padded_polygon

    def get_mitred_vertices(self, inflation_margin:float=None) -> NDArray[np.float32]:
        """
        Returns the coordinates of the corners of the shape of this boundary,
        padded by the robot radius (mitred padding)
        """
        if inflation_margin is None:
            inflation_margin = MobileRobot().cfg.RADIUS
        polygon = Polygon(self.vertices)
        return _exterior_nodes(polygon.buffer(-inflation_margin, join_style=JOIN_STYLE.mitre, mitre_limit=2), -1)

    def get_padded_vertices(self) -> NDArray[np.float32]:
        """
        Returns the coordinates of the corners of this boundary padded by the
        robot radius (round padding)
        """
        return _exterior_nodes(self.padded_polygon, -1)
    
    def collides(self, robot: MobileRobot) -> bool:
        """Whether ``robot`` collides with this boundary"""
        return not self.padded_polygon.contains(robot.point)
