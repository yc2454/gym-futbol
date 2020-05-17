import pymunk
from pymunk.vec2d import Vec2d


class Ball():

    def __init__(self, space, x, y, mass=10, radius=1, max_velocity=20,
                 elasticity=0.2):
        self.space = space
        self.max_velocity = max_velocity

        self.body, self.shape = self._setup_ball(
            space, x, y, mass, radius, elasticity)

    def get_position(self):
        x, y = self.body.position
        return [x, y]

    def get_velocity(self):
        vx, vy = self.body.velocity
        return [vx, vy]

    def get_observation(self):
        return self.get_position() + self.get_velocity()

    def set_position(self, x, y):
        self.body.position = x, y

    # apply force on the center of the player
    # fx: force in x direction
    # fy: force in y direction

    def apply_force_to_ball(self, fx, fy):
        # self.body.apply_force_at_local_point((fx,fy), point=(0, 0))
        self.body.apply_impulse_at_local_point((fx, fy), point=(0, 0))

    # return true if the ball and the player has contact.

    def has_contact_with(self, player):
        return self.shape.shapes_collide(player.shape).points != []

    def _setup_ball(self, space, x, y, mass, radius, elasticity):
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, moment)

        body.position = x, y
        body.start_position = Vec2d(body.position)

        def limit_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            l = body.velocity.length
            if l > self.max_velocity:
                scale = self.max_velocity / l
                body.velocity = body.velocity * scale

        body.velocity_func = limit_velocity

        shape = pymunk.Circle(body, radius)
        shape.color = (0, 1, 0, 1)  # green, (R,G,B,A)
        shape.elasticity = elasticity
        self.space.add(body, shape)

        return body, shape
