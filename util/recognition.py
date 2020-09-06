import numpy as np

import math
from math import degrees
from math import sqrt, acos, atan


def get_angle(a, b):
    del_y = a[1] - b[1]
    del_x = b[0] - a[0]
    if del_x == 0:
        del_x = 0.1

    angle = 0

    if del_x > 0 and del_y > 0:
        angle = degrees(atan(del_y / del_x))
    elif del_x < 0 and del_y > 0:
        angle = degrees(atan(del_y / del_x)) + 180

    return angle


def angle_gor(a, b, c, d):
    ab = [a[0] - b[0], a[1] - b[1]]
    ab1 = [c[0] - d[0], c[1] - d[1]]
    cos = abs(ab[0] * ab1[0] + ab[1] * ab1[1]) / (sqrt(ab[0] ** 2 + ab[1] ** 2) * sqrt(ab1[0] ** 2 + ab1[1] ** 2))
    ang = acos(cos)
    return ang * 180 / np.pi


def sit_ang(a, b, c, d):
    ang = angle_gor(a, b, c, d)
    s1 = 0
    if ang is not None:
        if ang < 120 and ang > 40:
            s1 = 1  
    return s1


def sit_rec(a, b, c, d):
    ab = [a[0] - b[0], a[1] - b[1]]
    ab1 = [c[0] - d[0], c[1] - d[1]]
    l1 = sqrt(ab[0] ** 2 + ab[1] ** 2)
    l2 = sqrt(ab1[0] ** 2 + ab1[1] ** 2)
    s = 0
    if l1 != 0 and l2 != 0:
        if (l2 / l1) >= 1.5:
            s = 1
    return s


def hip_knee_ankle_length(hip_pts, knee_pts, ankle_pts):
    knee_to_hip = [knee_pts[0] - hip_pts[0], knee_pts[1] - hip_pts[1]]
    knee_to_ankle = [knee_pts[0] - ankle_pts[0], knee_pts[1] - ankle_pts[1]]

    l1 = np.sqrt(np.dot(knee_to_hip, knee_to_hip))
    l2 = np.sqrt(np.dot(knee_to_ankle, knee_to_ankle))

    return l1, l2


def get_angle_knee_hip_ankle(knee_pts, hip_pts, ankle_pts):
    hip_x, hip_y = hip_pts[0], hip_pts[1]
    knee_x, knee_y = knee_pts[0], knee_pts[1]
    foot_x, foot_y = ankle_pts[0], ankle_pts[1]

    knee_hip_vec = (hip_x - knee_x, hip_y - knee_y)
    knee_foot_vec = (foot_x - knee_x, foot_y - knee_y)
    inner_product = np.dot(knee_hip_vec, knee_foot_vec)

    abs_knee_hip_vec = np.linalg.norm(knee_hip_vec)
    abs_knee_foot_vec = np.linalg.norm(knee_foot_vec)

    cosine_angle = inner_product / (abs_knee_hip_vec * abs_knee_foot_vec)
    
    radian_angle = np.arccos(cosine_angle)
    pi_angle = np.degrees(radian_angle)

    compare_pi_angle = 360 - pi_angle

    if compare_pi_angle < pi_angle:
        pi_angle = compare_pi_angle

    return pi_angle


if __name__ == "__main__":
    knee_pts = (0, 0)
    hip_pts = (-1, 10)
    foot_pts = (5, -10)

    angle = get_angle_knee_hip_foot(knee_pts, hip_pts, foot_pts)

    print(angle)


    # def getAngle(a, b, c):
    #     ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    #     return ang + 360 if ang < 0 else ang
    # # ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    
    # # ang + 360 if ang < 0 else ang
 
    # print(getAngle((5, 0), (0, 0), (0, 5)))