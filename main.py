# Import a library of functions called 'pygame'
import pygame
import numpy as np
import math

# Initialize the game engine
# pylint: disable=E1101
pygame.init()
size = [512, 512]


def get_rotate_x(theta):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotate_y(theta):
    return np.array(
        [
            [math.cos(theta), 0, math.sin(theta), 0],
            [0, 1, 0, 0],
            [-math.sin(theta), 0, math.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotate_z(theta):
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0, 0],
            [math.sin(theta), math.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def get_translated(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_np_arr(self, extra=False):
        if extra:
            return np.array([self.x, self.y, self.z, 1])
        return np.array([self.x, self.y, self.z])

    def translated(self, pos):
        x, y, z = pos
        return self.to_np_arr(True) @ get_translated(x, y, z)

    def rotate_x(self, theta):
        return self.to_np_arr(True) @ get_rotate_x(theta)

    def rotate_y(self, theta):
        return self.to_np_arr(True) @ get_rotate_y(theta)

    def rotate_z(self, theta):
        return self.to_np_arr(True) @ get_rotate_z(theta)

    def scale(self, n):
        return self.to_np_arr(True) @ np.array(
            [[n, 0, 0, 0], [0, n, 0, 0], [0, 0, n, 0], [0, 0, 0, 1]]
        )


class Line3D:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class TransformationStack:
    def __init__(self):
        self.stack = [np.identity(4)]

    def push(self):
        self.stack.append(self.stack[-1].copy())

    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            print("Transformation stack underflow")

    def get_current_matrix(self):
        return self.stack[-1]


class Camera:
    def __init__(self, position, yaw):
        self.position = position.to_np_arr(True)
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])
        self.h_fov = math.pi / 4
        self.v_fov = self.h_fov * (size[0] / size[1])
        self.near_plane = 0.01
        self.far_plane = 200
        self.moving_speed = 0.3
        self.rotation_speed = 0.015

        self.angle_pitch = 0
        self.angle_yaw = yaw
        self.angle_roll = 0

    def control(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_w]:
            self.position += self.forward * self.moving_speed
        if key[pygame.K_s]:
            self.position -= self.forward * self.moving_speed
        if key[pygame.K_a]:
            self.position -= self.right * self.moving_speed
        if key[pygame.K_d]:
            self.position += self.right * self.moving_speed
        if key[pygame.K_r]:
            self.position += self.up * self.moving_speed
        if key[pygame.K_f]:
            self.position -= self.up * self.moving_speed
        if key[pygame.K_q]:
            self.camera_yaw(self.rotation_speed)
        if key[pygame.K_e]:
            self.camera_yaw(-self.rotation_speed)
        if key[pygame.K_u]:
            print(self.position)
            print(self.angle_yaw)

    def camera_yaw(self, angle):
        self.angle_yaw += angle

    def camera_pitch(self, angle):
        self.angle_pitch += angle

    def axisIdentity(self):
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])

    def camera_update_axis(self):
        rotate = get_rotate_x(self.angle_pitch) @ get_rotate_y(self.angle_yaw)
        self.axisIdentity()
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate

        self.forward /= self.forward[3]
        self.right /= self.right[3]
        self.up /= self.up[3]

    def camera_matrix(self):
        self.camera_update_axis()
        return self.rotate_matrix() @ self.translate_matrix()

    def translate_matrix(self):
        x, y, z, w = self.position
        return np.array([[1, 0, 0, -x], [0, 1, 0, -y], [0, 0, 1, -z], [0, 0, 0, 1]])

    def rotate_matrix(self):
        e3 = ((self.right - self.position) / np.linalg.norm(self.right - self.position))
        e1 = ((e3 - self.up) / np.linalg.norm(e3 - self.up))
        e2 = ((e1 - e3) / np.linalg.norm(e1 - e3))
        rx, ry, rz, w = e1
        fx, fy, fz, w = e2
        ux, uy, uz, w = e3
        return np.array(
            [[rx, ry, rz, 0], [ux, uy, uz, 0], [fx, fy, fz, 0], [0, 0, 0, 1]]
        )


def loadHouse():
    house = []
    # Floor
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(5, 0, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 0, 5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(-5, 0, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 0, -5)))
    # Ceiling
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 5, -5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(5, 5, 5), Point3D(-5, 5, 5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(-5, 5, -5)))
    # Walls
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(-5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 5, 5)))
    # Door
    house.append(Line3D(Point3D(-1, 0, 5), Point3D(-1, 3, 5)))
    house.append(Line3D(Point3D(-1, 3, 5), Point3D(1, 3, 5)))
    house.append(Line3D(Point3D(1, 3, 5), Point3D(1, 0, 5)))
    # Roof
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(0, 8, -5)))
    house.append(Line3D(Point3D(0, 8, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(0, 8, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(0, 8, -5)))

    return house


def loadCar():
    car = []
    # Front Side
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-2, 3, 2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(2, 3, 2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(3, 2, 2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 1, 2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(-3, 1, 2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 2, 2)))

    # Back Side
    car.append(Line3D(Point3D(-3, 2, -2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(-2, 3, -2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, -2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 2, -2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(3, 1, -2), Point3D(-3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, -2), Point3D(-3, 2, -2)))

    # Connectors
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-3, 2, -2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 1, -2)))

    return car


def loadTire():
    tire = []
    # Front Side
    tire.append(Line3D(Point3D(-1, 0.5, 0.5), Point3D(-0.5, 1, 0.5)))
    tire.append(Line3D(Point3D(-0.5, 1, 0.5), Point3D(0.5, 1, 0.5)))
    tire.append(Line3D(Point3D(0.5, 1, 0.5), Point3D(1, 0.5, 0.5)))
    tire.append(Line3D(Point3D(1, 0.5, 0.5), Point3D(1, -0.5, 0.5)))
    tire.append(Line3D(Point3D(1, -0.5, 0.5), Point3D(0.5, -1, 0.5)))
    tire.append(Line3D(Point3D(0.5, -1, 0.5), Point3D(-0.5, -1, 0.5)))
    tire.append(Line3D(Point3D(-0.5, -1, 0.5), Point3D(-1, -0.5, 0.5)))
    tire.append(Line3D(Point3D(-1, -0.5, 0.5), Point3D(-1, 0.5, 0.5)))

    # Back Side
    tire.append(Line3D(Point3D(-1, 0.5, -0.5), Point3D(-0.5, 1, -0.5)))
    tire.append(Line3D(Point3D(-0.5, 1, -0.5), Point3D(0.5, 1, -0.5)))
    tire.append(Line3D(Point3D(0.5, 1, -0.5), Point3D(1, 0.5, -0.5)))
    tire.append(Line3D(Point3D(1, 0.5, -0.5), Point3D(1, -0.5, -0.5)))
    tire.append(Line3D(Point3D(1, -0.5, -0.5), Point3D(0.5, -1, -0.5)))
    tire.append(Line3D(Point3D(0.5, -1, -0.5), Point3D(-0.5, -1, -0.5)))
    tire.append(Line3D(Point3D(-0.5, -1, -0.5), Point3D(-1, -0.5, -0.5)))
    tire.append(Line3D(Point3D(-1, -0.5, -0.5), Point3D(-1, 0.5, -0.5)))

    # Connectors
    tire.append(Line3D(Point3D(-1, 0.5, 0.5), Point3D(-1, 0.5, -0.5)))
    tire.append(Line3D(Point3D(-0.5, 1, 0.5), Point3D(-0.5, 1, -0.5)))
    tire.append(Line3D(Point3D(0.5, 1, 0.5), Point3D(0.5, 1, -0.5)))
    tire.append(Line3D(Point3D(1, 0.5, 0.5), Point3D(1, 0.5, -0.5)))
    tire.append(Line3D(Point3D(1, -0.5, 0.5), Point3D(1, -0.5, -0.5)))
    tire.append(Line3D(Point3D(0.5, -1, 0.5), Point3D(0.5, -1, -0.5)))
    tire.append(Line3D(Point3D(-0.5, -1, 0.5), Point3D(-0.5, -1, -0.5)))
    tire.append(Line3D(Point3D(-1, -0.5, 0.5), Point3D(-1, -0.5, -0.5)))

    return tire


# Define the colors we will use in RGB format
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set the height and width of the screen
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Shape Drawing")

# Set needed variables
done = False
clock = pygame.time.Clock()
house_list = loadHouse()
car_list = loadCar()
tire_list = loadTire()

camera = Camera(Point3D(118.0, 25.3, 110.4), -29.15)
transformationStack = TransformationStack()

NEAR = camera.near_plane
FAR = camera.far_plane
RIGHT = math.tan(camera.h_fov / 2)
LEFT = -RIGHT
TOP = math.tan(camera.v_fov / 2)
BOTTOM = -TOP

zoomx = 2 / (RIGHT - LEFT)
zoomy = 2 / (TOP - BOTTOM)
n1 = (FAR + NEAR) / (FAR - NEAR)
n2 = -2 * NEAR * FAR / (FAR - NEAR)

wheel_rotation = 0
car_pos = 0

clip_matrix = np.array(
    [[zoomx, 0, 0, 0], [0, zoomy, 0, 0], [0, 0, n1, n2], [0, 0, 1, 0]]
)

to_screen_matrix = np.array(
    [
        [size[0] // 2, 0, size[0] // 2, 0],
        [0, -size[1] // 2, size[1] // 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


def pushDrawing(func, x=0, y=0, z=0, angle=0, rx=False, ry=False, rz=False):
    transformationStack.push()
    transformationStack.stack[-1] = transformationStack.stack[-1] @ get_translated(
        x, y, z
    )
    if rx:
        transformationStack.stack[-1] = transformationStack.stack[-1] @ get_rotate_x(
            angle
        )
    if ry:
        transformationStack.stack[-1] = transformationStack.stack[-1] @ get_rotate_y(
            angle
        )
    if rz:
        transformationStack.stack[-1] = transformationStack.stack[-1] @ get_rotate_z(
            angle
        )

    func()

    transformationStack.pop()


def drawObject(lines, color=BLUE):
    camera_matrix = camera.camera_matrix()

    for s in lines:
        # convert to 4D homogeneous coordinates
        start = s.start.to_np_arr(True)
        end = s.end.to_np_arr(True)

        # transformationStack.stack[-1] is using a matrix stack to put things from model space to world space
        # camera_matrix is the single matrix that converts from world to camera coordinates by using a translation and rotation matrix
        # clip_matrix is created at startup using the camera's near and far clipping planes and the field of view
        start_clipped = clip_matrix @ camera_matrix @ transformationStack.stack[-1] @ start
        end_clipped = clip_matrix @ camera_matrix @ transformationStack.stack[-1] @ end
        start_w = start_clipped[3]
        end_w = end_clipped[3]
        
        # apply perspective by dividing by w
        start_clipped /= start_w
        end_clipped /= end_w

        # # Check if both endpoints fail the view frustum test
        # if ((start_clipped[0] < -1 or end_clipped[0] < -1) and (start_clipped[0] > 1 or end_clipped[0] > 1)) \
        #         or ((start_clipped[1] < -1 or end_clipped[1] < -1) and (start_clipped[1] > 1 or end_clipped[1] > 1)) \
        #         or ((start_clipped[2] > 1 and end_clipped[2] > 1)):
        #     continue  # Reject the line
            
        # # Check if either endpoint fails the near plane test
        # if start_clipped[2] < -1 or end_clipped[2] < -1:
        #     continue  # Reject the line

        # to_screen_matrix is a viewport transformation using the screen size initialized at startup
        start_screen = start_clipped @ to_screen_matrix
        end_screen = end_clipped @ to_screen_matrix

        # drawing the line to the screen
        CENTER_SCREEN = 0 #size[0] / 2
        pygame.draw.line(
            screen,
            color,
            (start_screen[0] + CENTER_SCREEN, start_screen[1] + CENTER_SCREEN),
            (end_screen[0] + CENTER_SCREEN, end_screen[1] + CENTER_SCREEN),
        )


def drawHouse():
    drawObject(house_list, RED)


def drawCar():
    drawObject(car_list, GREEN)
    tire_spacing = 2
    pushDrawing(
        drawTire, tire_spacing, 0, tire_spacing, wheel_rotation, False, False, True
    )
    pushDrawing(
        drawTire, -tire_spacing, 0, tire_spacing, wheel_rotation, False, False, True
    )
    pushDrawing(
        drawTire, tire_spacing, 0, -tire_spacing, wheel_rotation, False, False, True
    )
    pushDrawing(
        drawTire, -tire_spacing, 0, -tire_spacing, wheel_rotation, False, False, True
    )


def drawTire():
    drawObject(tire_list)


time_since_last_action = 0

# Loop until the user clicks the close button.
while not done:
    # This limits the while loop to a max of 100 times per second.
    # Leave this out and we will use all CPU we can.
    dt = clock.tick(100)

    time_since_last_action += dt

    if time_since_last_action > 100:
        wheel_rotation += 0.03
        car_pos += 0.1
    # Clear the screen and set the screen background
    screen.fill(BLACK)

    # Controller Code#
    #####################################################################

    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # If user clicked close
            done = True

    camera.control()

    # Viewer Code#
    #####################################################################

    houseMargin = 20

    pushDrawing(drawHouse, 0, 0, houseMargin * 3, math.pi / 2, False, True)
    pushDrawing(drawHouse, 0, 0, houseMargin * 2, math.pi / 2, False, True)
    pushDrawing(drawHouse, 0, 0, houseMargin, math.pi / 2, False, True)
    pushDrawing(drawHouse, houseMargin, 0, 0)
    pushDrawing(drawHouse, houseMargin * 2, 0, 0)
    pushDrawing(drawHouse, houseMargin * 3, 0, houseMargin, -math.pi / 2, False, True)
    pushDrawing(
        drawHouse, houseMargin * 3, 0, houseMargin * 2, -math.pi / 2, False, True
    )
    pushDrawing(
        drawHouse, houseMargin * 3, 0, houseMargin * 3, -math.pi / 2, False, True
    )
    pushDrawing(drawCar, 40, 0, 40 + car_pos, math.pi / 2, False, True)

    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

# Be IDLE friendly
pygame.quit()
