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

from lib.extended_kalman_filter import ExtendedKalmanFilter

class threadMap(QThread):
    MapUpdate = pyqtSignal(bytes)

    def __init__(self, ip_address, port, update_speed_func, update_steer_func, queueList):
        super().__init__()
        self.ThreadActive = True
        self.initial_location = [10.87043, 106.80196]
        self.map = folium.Map(location=self.initial_location, zoom_start=17)
        # self.marker = folium.Marker(location=self.initial_location)
        # self.marker.add_to(self.map)
        self.PORT = port
        self.SERVER_ADDRESS = ip_address
        self.gps_coordinates = []
        self.ekf_coordinates = []
        self.update_speed_func = update_speed_func
        self.update_steer_func = update_steer_func
        self.connect_flag = False
        self.gps_latitude = 0.0
        self.gps_longitude = 0.0
        self.prev_gps_latitude = 0.0
        self.prev_gps_longtitude = 0.0
        self.heading = 0.0
        self.acc = 0.0
        self.steering_rate = 0.0
        self.queueList = queueList
        self.initial_flag = False
        self.wheelbase = 2.1
        self.length_rear = 1.0
        self.dt = 1
        # self.P_ekf = np.eye(5) * 0.1
        self.P_ekf = np.array([
            [0.54, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0],
            [0, 0, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0.1],
        ])
        # self.Q_ekf = np.eye(5) * 0.01
        self.Q_ekf = np.array([
            [0.54, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        # self.R_ekf = np.eye(5) * 0.1
        self.R_ekf = np.array([
            [0.54, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0.1],
        ])
        self.ekf = ExtendedKalmanFilter(self.wheelbase, self.length_rear, self.dt, self.P_ekf, self.Q_ekf, self.R_ekf)
        # Constants
        self.a = 6378137.0  # semi-major axis, meters
        self.f = 1 / 298.257223563  # flattening
        self.e2 = 2 * self.f - self.f ** 2  # eccentricity squared

        self.ref_lla = [10.87050, 106.802, 0]

    def lla_to_ecef(self, lat, lon, alt):
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat) ** 2)

        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (N * (1 - self.e2) + alt) * np.sin(lat)

        return np.array([X, Y, Z])

    def ecef_to_ned(self, ecef, ref_lla):
        ref_ecef = self.lla_to_ecef(*ref_lla)
        dX = ecef - ref_ecef

        lat, lon = np.deg2rad(ref_lla[0]), np.deg2rad(ref_lla[1])
        R = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                    [-np.sin(lon), np.cos(lon), 0],
                    [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])

        ned = R @ dX
        return ned

    def ned_to_ecef(self, ned, ref_lla):
        ref_ecef = self.lla_to_ecef(*ref_lla)
        lat, lon = np.deg2rad(ref_lla[0]), np.deg2rad(ref_lla[1])

        R = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                    [-np.sin(lon), np.cos(lon), 0],
                    [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])

        dX = np.linalg.inv(R) @ ned
        ecef = ref_ecef + dX

        return ecef

    def ecef_to_lla(self, ecef):
        X, Y, Z = ecef
        lon = np.arctan2(Y, X)

        p = np.sqrt(X ** 2 + Y ** 2)
        lat = np.arctan2(Z, p * (1 - self.e2))
        alt = 0

        for _ in range(5):
            N = self.a / np.sqrt(1 - self.e2 * np.sin(lat) ** 2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(Z, p * (1 - self.e2 * N / (N + alt)))

        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)

        return np.array([lat, lon, alt])

    def lla_to_ned(self, lla, ref_lla):
        ecef = self.lla_to_ecef(*lla)
        ned = self.ecef_to_ned(ecef, ref_lla)
        return ned

    def ned_to_lla(self, ned, ref_lla):
        ecef = self.ned_to_ecef(ned, ref_lla)
        lla = self.ecef_to_lla(ecef)
        return lla

    def run(self):
        self.ThreadActive = True
        # # Read GPS coordinates from CSV file
        # with open('route_points.csv', 'r') as file:
        #     csv_reader = csv.reader(file)
        #     next(csv_reader)  # Skip the header row
        #     coordinates = []

        #     for row in csv_reader:
        #         try:
        #             latitude, longitude = map(float, row)
        #             coordinates.append((latitude, longitude))
        #         except ValueError:
        #             print(f"Skipping invalid row: {row}")

        # # Create a PolyLine object with the coordinates
        # polyline = folium.PolyLine(locations=coordinates)

        # for i in range(len(coordinates)):
        #     if not self.ThreadActive:
        #         break

        #     # Add a PolyLine for the GPS points up to the current index
        #     folium.PolyLine(locations=coordinates[:i+1], color='blue').add_to(self.map)

        #     # Save map data to data object
        #     data = io.BytesIO()
        #     self.map.save(data, close_file=False)

        #     # Emit the map_updated signal with the updated map data
        #     self.MapUpdate.emit(data.getvalue())

        #     # Wait for 1 second before updating the next coordinate
        #     time.sleep(0.1)
        # self.terminate()

        ################ ESP SOCKET ############################
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Information Socket created')
        while not self.connect_flag:
            try:
                self.socket.connect((self.SERVER_ADDRESS, self.PORT))  # connect to the server
                self.connect_flag = True
            except:
                print('Connecting Failed!!! Retrying....')
                pass

        while self.ThreadActive:
            response_data = self.socket.recv(4*1024)
            try:
                response = pickle.loads(response_data)
                # response = json.load(response)
                print(f'Received response: {response}')
                self.gps_latitude = response["0"]
                self.gps_longitude = response["1"]
                # print('GPS lat: ', self.gps_latitude, "GPS long: ", self.gps_longitude)
                self.speed = response["2"]
                self.steer = response["3"]
                self.heading = response["4"]
                self.acc = response["5"]
                self.steering_rate = response["6"]
                # self.speed = random.randint(0,8)*3.6
                # self.steer = random.randint(-25, 25)
                # print('Speed: ', self.speed, 'Steer: ', self.steer)
                self.update_speed_func(self.speed*3.6)
                self.update_steer_func(self.steer)
                if not self.initial_flag:
                    if int(self.gps_latitude) != 0 and int(self.gps_longitude) != 0:
                        print(f'Received response: {response}')
                        self.marker = folium.Marker(location=[self.gps_latitude, self.gps_longitude])
                        self.marker.add_to(self.map)
                        self.gps_coordinates.append((self.gps_latitude, self.gps_longitude))
                        folium.PolyLine(locations=self.gps_coordinates, color='blue').add_to(self.map)
                        # Save map data to data object
                        data = io.BytesIO()
                        self.map.save(data, close_file=False)

                        # Emit the map_updated signal with the updated map data
                        self.MapUpdate.emit(data.getvalue())

                        # # Wait for 1 second before updating the next coordinate
                        # time.sleep(1)
                        print('Map Initialize Successfully!!!')
                        lat = self.gps_latitude
                        lon = self.gps_longitude
                        alt = 0.0
                        target_lla = [lat, lon, alt]
                        ned = self.lla_to_ned(target_lla, self.ref_lla)
                        self.current_state = [self.speed, self.steer, ned[0], ned[1], self.heading]  # [velocity, steering, x, y, heading]
                        self.current_covariance = self.P_ekf
                        # self.predicted_state, self.predicted_P = self.ekf.prediction_state(self.current_state, [self.acc, self.steering_rate], self.current_covariance, self.Q_ekf)
                        # self.measurement = np.array([self.speed, self.steer, ned[0], ned[1], self.heading])
                        # self.current_state, self.current_covariance = self.ekf.update_state(self.predicted_state, self.predicted_P, self.measurement, self.R_ekf)
                        self.initial_flag = True
                        self.prev_gps_latitude = self.gps_latitude
                        self.prev_gps_longtitude = self.gps_longtitude
                else:
                    if self.speed != 0:
                        self.predicted_state, self.predicted_P = self.ekf.prediction_state(self.current_state, [self.acc, self.steering_rate], self.current_covariance, self.Q_ekf)
                    if int(self.gps_latitude) != 0 and int(self.gps_longitude) != 0:
                        print(f'Received response: {response}')
                        self.gps_coordinates.append((self.gps_latitude, self.gps_longitude))
                        folium.PolyLine(locations=self.gps_coordinates, color='blue').add_to(self.map)

                        # self.current_covariance = self.P_ekf
                        if self.speed != 0 and self.prev_gps_latitude != self.gps_latitude and self.prev_gps_longtitude != self.gps_longtitude:
                            lat = self.gps_latitude
                            lon = self.gps_longitude
                            alt = 0.0
                            target_lla = [lat, lon, alt]
                            ned = self.lla_to_ned(target_lla, self.ref_lla)
                            self.measurement = np.array([self.speed, self.steer, ned[0], ned[1], self.heading])
                            self.current_state, self.current_covariance = self.ekf.update_state_gps(self.predicted_state, self.predicted_P, self.measurement, self.R_ekf)
                            ned = [self.current_state[2], self.current_state[3], 0]
                            lla = self.ned_to_lla(ned, self.ref_lla)
                            self.ekf_coordinates.append((lla[0], lla[1]))
                            folium.PolyLine(locations=self.ekf_coordinates, color='red').add_to(self.map)
                            # time.sleep(0.2)
                            self.prev_gps_latitude = self.gps_latitude
                            self.prev_gps_longtitude = self.gps_longtitude
                        elif self.speed != 0:
                            self.measurement = np.array([self.speed, self.steer, 0, 0, self.heading])
                            self.current_state, self.current_covariance = self.ekf.update_state_vel_head(self.predicted_state, self.predicted_P, self.measurement, self.R_ekf)

                        # Save map data to data object
                        data = io.BytesIO()
                        self.map.save(data, close_file=False)

                        # Emit the map_updated signal with the updated map data
                        self.MapUpdate.emit(data.getvalue())

                    # # Wait for 1 second before updating the next coordinate
                    # time.sleep(1)
                # time.sleep(1)
            except:
                self.update_speed_func(self.speed*3.6)
                self.update_steer_func(self.steer)
                folium.PolyLine(locations=self.gps_coordinates, color='blue').add_to(self.map)
                folium.PolyLine(locations=self.ekf_coordinates, color='red').add_to(self.map)
                # Save map data to data object
                data = io.BytesIO()
                self.map.save(data, close_file=False)

                # Emit the map_updated signal with the updated map data
                self.MapUpdate.emit(data.getvalue())
            
            time.sleep(1)
        self.socket.close()
        self.terminate()

    def stop(self):
        self.ThreadActive = False
        self.socket.close()
        self.terminate()
