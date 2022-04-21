import math

from gym import spaces
from gym_miniworld.entity import Ball, Box, Key, MeshEnt
from gym_miniworld.miniworld import MiniWorldEnv

TEXTURES = [
    "asphalt",
    "brick_wall",
    "cardboard",
    "ceiling_tile_noborder",
    "ceiling_tiles",
    "cinder_blocks",
    "concrete",
    "concrete_tiles",
    "drywall",
    "floor_tiles_bw",
    "grass",
    "lava",
    "marble",
    "metal_grill",
    "rock",
    "slime",
    "stucco",
    "water",
    "wood",
    "wood_planks",
]


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

    SIZE = 10

    def __init__(self, max_angle=2 * math.pi, num_objs=6, **kwargs):
        self.num_objs = num_objs

        if max_angle == 2 * math.pi:  # For a full circle
            # Distribute across [0, 2pi[, right-open interval, so items don't overlay
            self.revolution_step = max_angle / self.num_objs
        elif max_angle < 2 * math.pi:  # For a partial circle
            # Distribute across [0, max_angle], right-closed interval
            self.revolution_step = max_angle / (self.num_objs - 1)
        else:
            raise ValueError("max_angle must be lower or equal 2pi")

        super().__init__(max_episode_steps=math.inf, **kwargs)

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.turn_right + 1)

    def _gen_world(self):
        floor_tex = self.rand.choice(TEXTURES) if self.domain_rand else "asphalt"
        wall_tex = self.rand.choice(TEXTURES) if self.domain_rand else "brick_wall"
        ceil_tex = self.rand.choice(TEXTURES) if self.domain_rand else "concrete_tiles"
        no_ceiling = self.rand.bool() if self.domain_rand else True

        self.add_rect_room(
            min_x=-self.SIZE // 2,
            max_x=self.SIZE // 2,
            min_z=-self.SIZE // 2,
            max_z=self.SIZE // 2,
            floor_tex=floor_tex,
            wall_tex=wall_tex,
            ceil_tex=ceil_tex,
            no_ceiling=no_ceiling,
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
