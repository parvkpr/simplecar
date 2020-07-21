import time

import random
import math
import numpy as np

from gym import spaces

import pyglet
from pyglet.window import Window

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pyglet_util

COLLTYPE_OBS = 5
COLLTYPE_ACT = 1
COLLTYPE_EDGE = 7

def colls(space, arbiter, a):
    global COL
    COL = True
    pyglet.app.exit()
    return True

class Env:

    def __init__(self, screen_width=1000, screen_height=1200, update_fps=45, refresh_fps=15, init_pos=(200, 500),
                 seed=10992, goal=(800, 700), threshold=1):
        self.screen = Window(width=screen_width, height=screen_height, vsync=False)
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.max_velocity = 30
        self.steering = 0
        self.space = pymunk.Space()
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.update_fps = update_fps
        self.refresh_fps = refresh_fps
        self.space.iterations = 10
        self.space.sleep_time_threshold = 0.5
        self.static_body = self.space.static_body
        shape = pymunk.Segment(self.static_body, (1, 1), (1, screen_height), 1.0)
        self.space.add(shape)
        shape.collision_type = COLLTYPE_EDGE
        shape.elasticity = 1
        shape.friction = 1
        self.action_space = spaces.Box(low=np.array([-5, -5]), high=np.array([5, 5]))
        self.state_space = spaces.Box(low=np.array([-self.max_velocity, -self.max_velocity, 0, 0, 0]),
                                            high=np.array(
                                                [self.max_velocity, self.max_velocity, screen_width, screen_height,
                                                 360]))
        global COL
        COL = False
        self.init_pos = init_pos
        shape = pymunk.Segment(self.static_body, (screen_width, 1), (screen_width, screen_height), 1.0)
        self.space.add(shape)
        shape.elasticity = 1
        shape.friction = 1
        shape.collision_type = COLLTYPE_EDGE
        self.seed = seed
        random.seed(seed)
        self.goal = goal
        self.threshold = threshold
        shape = pymunk.Segment(self.static_body, (1, 1), (screen_width, 1), 1.0)
        self.space.add(shape)
        shape.elasticity = 1
        shape.friction = 1
        shape.collision_type = COLLTYPE_EDGE
        self.goalcol = 3
        shape = pymunk.Segment(self.static_body, (1, screen_height), (screen_width, screen_height), 1.0)
        self.space.add(shape)
        shape.collision_type = COLLTYPE_EDGE
        shape.elasticity = 1
        shape.friction = 1
        self.add_obs(2, 50)
        self.add_actor(init_pos)
        h = self.space.add_collision_handler(COLLTYPE_OBS, COLLTYPE_ACT)
        h.begin = colls
        f = self.space.add_collision_handler(COLLTYPE_EDGE, COLLTYPE_ACT)
        f.begin = colls

    def add_box(self, size, mass, type = 'OBS'):
        radius = Vec2d(size, size).length

        body = pymunk.Body()
        self.space.add(body)

        body.position = Vec2d(
            random.random() * (self.screen_width - 2 * radius) + radius,
            random.random() * (self.screen_height - 2 * radius) + radius
        )

        self.shape = pymunk.Poly.create_box(body, (size, size), 0.0)
        self.shape.mass = mass
        self.shape.friction = 0.7
        self.space.add(self.shape)
        if type == 'OBS':
            self.shape.collision_type = COLLTYPE_OBS
        else:
            self.shape.collision_type = COLLTYPE_ACT
            self.shape.filter = pymunk.ShapeFilter(categories=0b1)

        return body

    def goal_setup(self):
        goal_body = pymunk.Body()
        goal_body.position = self.goal
        goal = pymunk.Circle(goal_body, self.threshold)
        goal.collision_type = self.goalcol
        self.space.add(goal)

    def add_obs(self, num, size):
        for _ in range(num):
            body = self.add_box(size, 1)

            pivot = pymunk.PivotJoint(self.static_body, body, (0, 0), (0, 0))
            self.space.add(pivot)
            pivot.max_bias = 0  # disable joint correction
            pivot.max_Force = 1000  # emulate linear friction

            gear = pymunk.GearJoint(self.static_body, body, 0.0, 1.0)
            self.space.add(gear)
            gear.max_bias = 0  # disable joint correction
            gear.max_force = 5000  # emulate angular friction

    def add_actor(self, position=(0, 0)):
        self.actor_c_b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.actor_c_b.position = position
        self.space.add(self.actor_c_b)
        self.actor_b = self.add_box(20, 1, 'ACT')
        self.actor_b.position = position

        for s in self.actor_b.shapes:
            s.color = (0, 100, 100, 255)

        pivot = pymunk.PivotJoint(self.actor_c_b, self.actor_b, (0, 0), (0, 0))
        self.space.add(pivot)
        pivot.max_bias = 0
        pivot.max_force = 10000

        gear = pymunk.GearJoint(self.actor_c_b, self.actor_b, 0.0, 0.1)
        self.space.add(gear)
        gear.error_bias = 0
        gear.max_bias = 1.2
        gear.max_force = 50000

        pin = pymunk.SlideJoint(self.actor_c_b, self.actor_b, (0,0), (0,0), 0.5, 1)
        self.space.add(pin)

    def refresh(self, dt):
        self.actor_c_b.force = (0,0)
        self.steering = 0

    def state(self):
        state_val = np.array([ self.actor_b.velocity.x, self.actor_b.velocity.y,  self.actor_b.position.x,  self.actor_b.position.y, self.actor_b.angle])
        return state_val

    def step(self, throttle):
        self.actor_c_b.force = throttle
        run(self)
        dist = get_dist(self.goal, self.actor_b.position)
        if dist <= self.threshold:
            self.done = True
        else:
            self.done = False
        reward = 0
        global COL
        if COL is True:
            self.done = True
        filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS ^ 0b1)
        dist_to_obstacle = self.space.point_query_nearest(self.actor_b.position, 1000, filter)
        print(dist_to_obstacle)
        return self.state(),reward, self.done, None

    def reset(self):
        self.actor_c_b.force = (0,0)
        self.steering = 0
        self.actor_c_b.position = self.init_pos
        self.actor_c_b.velocity = (0,0)
        self.actor_b.position = self.init_pos
        self.actor_b.velocity = (0,0)
        self.actor_b.force = (0,0)
        self.actor_c_b.angle = 0
        self.actor_b.angle = 0
        return self.state()

    def update(self, dt):
        # Update the car kinematics

        self.actor_c_b.velocity += (self.actor_c_b.force/self.actor_b.mass) * dt
        self.actor_c_b.velocity.x = max(-self.max_velocity,
                              min(self.actor_c_b.velocity.x, self.max_velocity))
        self.space.step(dt)





def get_dist(a, b):
    dist = (a[0] - b[0])**2 + (a[1] - b[1])**2
    return math.sqrt(dist)


def run(e = None):

    if e is None:
        e = Env()

    pyglet.clock.schedule_interval(e.update, 1 / e.update_fps)
    pyglet.clock.schedule_interval(e.refresh, 1 / e.refresh_fps)
    window = e.screen


    @window.event
    def on_draw():
        pyglet.gl.glClearColor(255, 255, 255, 255)
        window.clear()
        e.space.debug_draw(e.draw_options)
        time_end = time.time()
        if (time_end - time_beg) > 1:
            pyglet.app.exit()

    global time_beg
    time_beg = time.time()

    pyglet.app.run()

    return e


if __name__=='__main__':
    e = Env()
    pyglet.clock.schedule_interval(e.update, 1 / e.update_fps)
    pyglet.clock.schedule_interval(e.refresh, 1 / e.refresh_fps)
    window = e.screen


    @window.event
    def on_draw():
        pyglet.gl.glClearColor(255, 255, 255, 255)
        window.clear()
        e.space.debug_draw(e.draw_options)

    pyglet.app.run()

# usage
# from unicyclev2 import Env, run
# if __name__ == '__main__':
#     e = Env()
#     run(e)
#     e.reset()
#     for i in range(20):
#         ac = e.action_space.sample()
#         print(ac)
#         print(e.step(ac))
#         print('executing now' + repr(i))
