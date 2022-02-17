import math

from gym import spaces
from gym_miniworld.entity import Ball, Box, Key, MeshEnt
from gym_miniworld.miniworld import MiniWorldEnv


class Barrel(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="barrel", height=size, static=True)


class Chair(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="office_chair", height=size, static=True)


class Cone(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="cone", height=size, static=True)


class Duckie(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="duckie", height=size, static=True)


class KeyCard(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="keycard", height=size, static=True)


class MedKit(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="medkit", height=size, static=True)


class Potion(MeshEnt):
    def __init__(self, size=0.6):
        super().__init__(mesh_name="potion", height=size, static=True)


class LookAtObjs(MiniWorldEnv):
    """
    Room with multiple objects in which the agent can only turn around.
    """

    def __init__(self, num_objs=5, **kwargs):
        self.size = 10
        self.num_objs = num_objs
        self.revolution_step = (2 * math.pi) / self.num_objs

        super().__init__(max_episode_steps=math.inf, **kwargs)

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.turn_right + 1)

    def _gen_world(self):
        self.add_rect_room(
            min_x=-self.size // 2,
            max_x=self.size // 2,
            min_z=-self.size // 2,
            max_z=self.size // 2,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        objs_type = self.rand.subset(
            [Ball, Barrel, Box, Chair, Cone, Duckie, Key, KeyCard, MedKit, Potion],
            self.num_objs,
        )
        for i, obj_type in enumerate(objs_type):
            pos = [
                1.5 * math.cos(self.revolution_step * i),
                1,
                1.5 * math.sin(self.revolution_step * i),
            ]

            if obj_type == Box or obj_type == Ball or obj_type == Key:
                self.place_entity(obj_type(color=self.rand.color()), pos=pos)
            else:
                self.place_entity(obj_type(), pos=pos)

        self.place_entity(self.agent, pos=[0, 0, 0])
