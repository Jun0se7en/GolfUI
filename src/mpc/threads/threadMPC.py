import cv2
import threading
import socket
import base64
import time
import numpy as np
import os
import io
import csv
import json
import random

from multiprocessing import Pipe
from src.templates.threadwithstop import ThreadWithStop
import struct
import pickle
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsColorizeEffect
from PyQt5.QtGui import QFontDatabase, QPixmap, QImage
from PyQt5.QtCore import QThread, QEvent, pyqtSignal

import folium
from ui_interface import *


class threadMPC(QThread):
    MapUpdate = pyqtSignal(bytes)

    def __init__(self, queuesList):
        super().__init__()
        self.queuesList = queuesList
        self.ThreadActive = True
        # Thông số xe và trạng thái ban đầu
        self.wheelbase = 2.5
        self.length_rear = 1.0
        self.dt = 0.1
        self.horizon = 10  # Horizon để MPC nhìn về tương lai
        self.Q_mpc = np.array([1.0, 1.0, 1.0])  # Trọng số cho các trạng thái [x, y, heading]
        self.R_mpc = np.array([0.1, 0.1])  # Trọng số cho các tín hiệu điều khiển [velocity, steering_angle]
        self.P_init = np.eye(5) * 0.1
        self.Q_ekf = np.eye(5) * 0.01
        self.R_ekf = np.eye(5) * 0.1  # Chỉ cần 4 trạng thái trong đo lường (velocity, x, y, heading)
        self.learning_rate = 5  # tốc độ học của thuật toán gradient descent.
        self.max_iter = 10 # số lần lặp tối đa của thuật toán gradient descent.
        self.beta1 = 0.9 # Hệ số momentum đầu tiên của Adam.S
        self.beta2 = 0.999 # Hệ số momentum 2 của Adam.
        self.epsilon = 1e-8 #Giá trị rất nhỏ để tránh chia cho 0 trong quá trình tính toán của Adam.
        self.lip_value = 1.0 #Giá trị giới hạn gradient.
        # Tạo đối tượng EKF
        self.ekf = ExtendedKalmanFilter(self.wheelbase, self.length_rear, self.dt, self.P_init, self.Q_ekf, self.R_ekf)

        # Tạo đối tượng MPC
        self.mpc = ModelPredictiveControl(self.wheelbase, self.length_rear, self.dt, self.horizon, self.Q_mpc, self.R_mpc, self.ekf,self.learning_rate,self.max_iter,self.beta1,self.beta2,self.epsilon,self.clip_value)
        # Tạo đường tham chiếu
        path = pd.read_csv('./route_points.csv')
        self.reference_x, self.reference_y = path['latitude'].tolist(), path['longtitude'].tolist()
        self.num_points = len(self.reference_x)
        

        # Trạng thái ban đầu
        current_state = [1.0, 0.0, reference_x[0], reference_y[0], 0.0]  # [velocity, steering, x, y, heading]
        current_covariance = P_init

       # Mô phỏng chuyển động của xe trong một số bước
        predicted_trajectory = np.zeros((num_points, 2))
        real_trajectory = np.zeros((num_points, 2))  # Thêm quỹ đạo thực tế
        control_inputs = []
        i = 0
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            for t in range(horizon):
                idx = (i + t) % num_points
                reference.append([0.0, 0.0, reference_x[idx], reference_y[idx], 0.0])

            velocity, steering_angle, predicted_state, predicted_covariance = mpc.control(current_state, reference, current_covariance, Q_ekf)
            control_inputs.append([velocity, steering_angle])

            measurement_noise = np.random.uniform(-0.5, 0.5, 5)
            measurement = np.array([predicted_state[0], predicted_state[1],predicted_state[2], predicted_state[3], predicted_state[4]])  + measurement_noise

            current_state, current_covariance = mpc.update_with_measurement(predicted_state, predicted_covariance, measurement, R_ekf)

            real_trajectory[i, 0] = current_state[2]
            real_trajectory[i, 1] = current_state[3]
            predicted_trajectory[i, 0] = predicted_state[2]
            predicted_trajectory[i, 1] = predicted_state[3]

            predicted_horizon_trajectory = mpc.predict_trajectory(current_state, [velocity, steering_angle] * horizon)
            predicted_horizon_trajectory = np.array(predicted_horizon_trajectory)
            i += 1
        self.terminate()

    def stop(self):
        self.ThreadActive = False
        self.socket.close()
        self.terminate()
