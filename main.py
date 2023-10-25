# Import a library of functions called 'pygame'
import pygame
import numpy as np
import math

# Initialize the game engine
# pylint: disable=E1101
pygame.init()
size = [512, 512]

def get_rotate_x(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(theta), math.sin(theta), 0],
        [0, -math.sin(theta), math.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def get_rotate_y(theta): 
    return np.array([
        [math.cos(theta), 0, -math.sin(theta), 0],
        [0, 1, 0, 0],
        [math.sin(theta), 0, math.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def get_rotate_z(theta):
    return np.array([
        [math.cos(theta), math.sin(theta), 0, 0],
        [-math.sin(theta), math.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def get_translated(x, y, z):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [x, y, z, 1]
    ])

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

    def rotate_y (self, theta):
        return self.to_np_arr(True) @ get_rotate_y(theta)
    
    def rotate_z(self, theta):
        return self.to_np_arr(True) @ get_rotate_z(theta)
    
    def scale(self, n):
        return self.to_np_arr(True) @ np.array([
            [n, 0, 0, 0],
            [0, n, 0, 0],
            [0, 0, n, 0],
            [0, 0, 0, 1]
        ])

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
        self.up = np.array([0, -1, 0, 1])
        self.right = np.array([1, 0, 0, 1])
        self.h_fov = math.pi / 4
        self.v_fov = self.h_fov * (size[0] / size[1])
        self.near_plane = 0.1
        self.far_plane = 10
        self.moving_speed = 0.3
        self.rotation_speed = 0.015

        self.angle_pitch = 0
        self.angle_yaw = yaw
        self.angle_roll = 0

    def control(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_w]:
            self.position -= self.forward * self.moving_speed
        if key[pygame.K_s]:
            self.position += self.forward * self.moving_speed
        if key[pygame.K_a]:
            self.position += self.right * self.moving_speed
        if key[pygame.K_d]:
            self.position -= self.right * self.moving_speed
        if key[pygame.K_r]:
            self.position -= self.up * self.moving_speed
        if key[pygame.K_f]:
            self.position += self.up * self.moving_speed
        if key[pygame.K_q]:
            self.camera_yaw(-self.rotation_speed)
        if key[pygame.K_e]:
            self.camera_yaw(self.rotation_speed)
        if key[pygame.K_u]:
            print(self.position)
            print(self.angle_yaw)

    def camera_yaw(self, angle):
        self.angle_yaw += angle

    def camera_pitch(self, angle):
        self.angle_pitch += angle

    def axisIdentity(self):
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, -1, 0, 1])
        self.right = np.array([1, 0, 0, 1])

    def camera_update_axis(self):
        rotate = get_rotate_x(self.angle_pitch) @ get_rotate_y(self.angle_yaw)
        self.axisIdentity()
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate

    def camera_matrix(self):
        self.camera_update_axis()
        return self.translate_matrix() @ self.rotate_matrix()

    def translate_matrix(self):
        x, y, z, w = self.position
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-x, -y, -z, 1]
        ])

    def rotate_matrix(self):
        rx, ry, rz, w = self.right
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        return np.array([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0, 0, 0, 1]
        ])


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

camera = Camera(Point3D(105.0, 40.0, 105.0), 0.40079632679489985)
transformationStack = TransformationStack()

NEAR = camera.near_plane
FAR = camera.far_plane
RIGHT = math.tan(camera.h_fov / 2)
LEFT = -RIGHT
TOP = math.tan(camera.v_fov / 2)
BOTTOM = -TOP

m00 = 2 / (RIGHT - LEFT)
m11 = 2 / (TOP - BOTTOM)
m22 = (FAR + NEAR) / (FAR - NEAR)
m32 = -2 * NEAR * FAR / (FAR - NEAR)

wheel_rotation = 0
car_pos = 0

projection_matrix = np.array([
    [m00, 0, 0, 0],
    [0, m11, 0, 0],
    [0, 0, m22, 1],
    [0, 0, m32, 0]
])

to_screen_matrix = np.array([
    [size[0] // 2, 0, size[0] // 2, 0],
    [0, -size[1] // 2, size[1] // 2, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

def pushDrawing(func, x=0, y=0, z=0, angle=0, rx=False, ry=False, rz=False): 
    transformationStack.push()
    transformationStack.stack[-1] = get_translated(x, y, z) @ transformationStack.stack[-1]
    if rx:
        transformationStack.stack[-1] = get_rotate_x(angle) @ transformationStack.stack[-1]
    if ry:
        transformationStack.stack[-1] = get_rotate_y(angle) @ transformationStack.stack[-1]
    if rz:
        transformationStack.stack[-1] = get_rotate_z(angle) @ transformationStack.stack[-1]
    
    func()

    transformationStack.pop()

def drawObject(lines, color=BLUE):
    camera_matrix = camera.camera_matrix()
    
    for s in lines:
        start = s.start.to_np_arr(True)
        end = s.end.to_np_arr(True)
        start_projected = start @ transformationStack.stack[-1] @ camera_matrix @ projection_matrix
        end_projected = end @ transformationStack.stack[-1] @ camera_matrix @ projection_matrix

        start_projected /= start_projected[3]
        end_projected /= end_projected[3]

        start_screen = start_projected @ to_screen_matrix 
        end_screen = end_projected @ to_screen_matrix 

        # if (start_screen[0] > 0 and start_screen[1] > 0 and end_screen[0] > 0 and end_screen[1] > 0):
        pygame.draw.line(screen, color, (start_screen[0], start_screen[1]), (end_screen[0], end_screen[1]))

def drawHouse():
    drawObject(house_list, RED)

def drawCar():
    drawObject(car_list, GREEN)
    tire_spacing = 2
    pushDrawing(drawTire, tire_spacing, 0, tire_spacing, wheel_rotation, False, False, True)
    pushDrawing(drawTire, -tire_spacing, 0, tire_spacing, wheel_rotation, False, False, True)
    pushDrawing(drawTire, tire_spacing, 0, -tire_spacing, wheel_rotation, False, False, True)
    pushDrawing(drawTire, -tire_spacing, 0, -tire_spacing, wheel_rotation, False, False, True)

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

    # drawObject(house_list)
    pushDrawing(drawHouse, 0, 0, houseMargin * 3, math.pi/2, False, True)
    pushDrawing(drawHouse, 0, 0, houseMargin * 2, math.pi/2, False, True)
    pushDrawing(drawHouse, 0, 0, houseMargin, math.pi/2, False, True)
    pushDrawing(drawHouse, houseMargin, 0, 0)
    pushDrawing(drawHouse, houseMargin * 2, 0, 0)
    pushDrawing(drawHouse, houseMargin * 3, 0, houseMargin, -math.pi/2, False, True)
    pushDrawing(drawHouse, houseMargin * 3, 0, houseMargin * 2, -math.pi/2, False, True)
    pushDrawing(drawHouse, houseMargin * 3, 0, houseMargin * 3, -math.pi/2, False, True)
    pushDrawing(drawCar, 40, 0, 40 + car_pos, math.pi/2, False, True)

    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

# Be IDLE friendly
pygame.quit()
