"""
Face Find Apppication for Facial Analysis
"""

from fbs_runtime.application_context import ApplicationContext
# import libs for the interface
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
# import libs for the model detections
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
import cv2
import sys
# import libs for data collection and processing
import matplotlib.pyplot as plt
from statistics import mode
import datetime as dt
import numpy as np
import time
import math
import mss
import csv
import os
# extras; resizing and popup import
import imutils
from popupDialog import Ui_Dialog

class UiMainWindow():
    def __init__(self):
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.tab_widget = QtWidgets.QTabWidget(self.centralwidget)
        self.tab = QtWidgets.QWidget()
        self.display_user = QtWidgets.QLabel(self.tab)
        self.tab_2 = QtWidgets.QWidget()
        self.emotion_stats = QtWidgets.QLabel(self.tab_2)
        self.tab_3 = QtWidgets.QWidget()
        self.head_pose_stats = QtWidgets.QLabel(self.tab_3)
        self.tab_4 = QtWidgets.QWidget()
        self.recording_stats = QtWidgets.QLabel(self.tab_4)
        self.splitter_2 = QtWidgets.QSplitter(self.centralwidget)
        self.check_box = QtWidgets.QCheckBox(self.centralwidget)
        self.top_label = QtWidgets.QLabel(self.centralwidget)
        self.tool_button = QtWidgets.QToolButton(self.centralwidget)
        self.top_label2 = QtWidgets.QLabel(self.centralwidget)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.record_btn = QtWidgets.QPushButton(self.splitter)
        self.menubar = QtWidgets.QMenuBar(main_window)
        self.statusbar = QtWidgets.QStatusBar(main_window)

        # initialize the capture of the frames along with the size of the capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 551)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 921)

        # start the clock timer for refreshing the frames
        self.timer = QTimer(main_window)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5) 

        if os.path.exists(os.path.join(os.path.expanduser('~'), 'Downloads', 'FYP_Data')):
            pass
        else:
            os.mkdir(os.path.join(os.path.expanduser('~'), 'Downloads', 'FYP_Data'))

        # temp variable for controlling the popup
        self.temp = False

        # detect faces
        self.detector = dlib.get_frontal_face_detector()
        self.face_enabled = False

        # detect head orientation
        landmark_model = "trained_models/shape_predictor_68_face_landmarks.dat"
        self.engagement_canvas = np.zeros((270, 300, 3), dtype="uint8")
        self.predictor = dlib.shape_predictor(landmark_model)
        self.pos = ["Pitch", "Roll", "Yaw"]

        # detect emotions
        self.color = (0, 0, 255)
        self.emotion_model_path = 'trained_models/fer2013_big_XCEPTION.54-0.66.hdf5'
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprised", "Neutral"]

        self.emotion_labels = {0: 'Angry',
                               1: 'Disgust',
                               2: 'Fear',
                               3: 'Happy',
                               4: 'Sad',
                               5: 'Surprise',
                               6: 'Neutral'}
        self.emotion_window = []
        self.emotion_canvas = np.zeros((270, 300, 3), dtype="uint8")

        # calculation for Euler Angle
        self.model_points = np.float32([[6.825897, 6.760612, 4.402142], # left eyebrow: left
                                        [1.330353, 7.122144, 6.903745], # left eyebrow: right
                                        [-1.330353, 7.122144, 6.903745], # right eyebrow: left
                                        [-6.825897, 6.760612, 4.402142],  # right eyebrow: right
                                        [5.311432, 5.485328, 3.987654], # left eye: left corner
                                        [1.789930, 5.393625, 4.413414], # left eye: right corner
                                        [-1.789930, 5.393625, 4.413414], # right eye: left corner
                                        [-5.311432, 5.485328, 3.987654], # right eye: right corner
                                        [2.005628, 1.409845, 6.165652], # nose left
                                        [0.000000, 0.000000, 10.000000], # nose tip
                                        [-2.005628, 1.409845, 6.165652], # nose right
                                        [2.774015, -2.080775, 5.048531], # mouth left
                                        [-2.774015, -2.080775, 5.048531], #mouth right
                                        [0.000000, -3.116408, 6.097667], # mouth bottom
                                        [0.000000, -7.415691, 4.070434]]) # chin

        # data collection
        self.feeling_angry = 0
        self.feeling_sad = 0
        self.feeling_neutral = 0
        self.feeling_happy = 0
        self.feeling_surprised = 0
        self.feeling_fear = 0
        self.feeling_disgust = 0
        self.is_engaged = 0
        self.is_not_engaged = 0

        # get the new ID number for the next candidate
        self.next_candidate_id = self.get_candidate_id()

        # register the recordings start time
        self.starttime = dt.datetime.now().strftime("%H:%M:%S")

        # register the recordings start time
        self.starttime = dt.datetime.now().strftime("%H:%M:%S")

        # fix for looping
        # self.controller = True

        # initialize the frame counter and the arrays for data collection
        self.frame_counter = 0
        self.engaged_array = []
        self.emotion_array = []
        self.capture_time_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.capture_time_2 = [0, 0, 0, 0]
        self.appended_emotion_array = ['',
                                       'Absent',
                                       'Angry',
                                       'Sad',
                                       'Neutral',
                                       'Happy',
                                       'Surprise',
                                       'Fear',
                                       'Disgust',
                                       '']
        self.appended_engagement_array = ['Not Measured',
                                          'Engaged',
                                          'Not Engaged',
                                          '']
        self.last_emotion_tracked = "Neutral"

        # initialise the screen recording
        self.sct = mss.mss()
        self.bbox = (100, 10, 1680, 1050)

        # control variables for the recording
        self.set_fps_to_record = True
        self.grab_frame_two = 0

        # register the recordings end time
        self.endtime = dt.datetime.now().strftime("%H:%M:%S")

        self.headpose_canvas = np.zeros((270, 300), dtype="uint8")
        self.emotion_canvas = np.zeros((270, 300), dtype="uint8")


    def setup_ui(self, main_window):
        main_window.setObjectName("main_window")
        main_window.setEnabled(True)
        main_window.resize(751, 685)
        main_window.setMinimumSize(QtCore.QSize(751, 685))
        main_window.setMaximumSize(QtCore.QSize(751, 685))
        self.centralwidget.setObjectName("centralwidget")
        self.tab_widget.setGeometry(QtCore.QRect(80, 100, 581, 401))
        self.tab_widget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.tab_widget.setStyleSheet("background-color:white; color:black;")
        self.tab_widget.setObjectName("tab_widget")
        self.tab.setAutoFillBackground(False)
        self.tab.setObjectName("tab")
        self.display_user.setGeometry(QtCore.QRect(0, -10, 581, 401))
        self.display_user.setStyleSheet("background-color:#919191;")
        self.display_user.setText("")
        self.display_user.setObjectName("display_user")
        self.tab_widget.addTab(self.tab, "")
        self.tab_2.setObjectName("tab_2")
        self.emotion_stats.setGeometry(QtCore.QRect(0, 0, 581, 401))
        self.emotion_stats.setStyleSheet("background-color:#919191;")
        self.emotion_stats.setText("")
        self.emotion_stats.setObjectName("emotion_stats")
        self.tab_widget.addTab(self.tab_2, "")
        self.tab_3.setObjectName("tab_3")
        self.head_pose_stats.setGeometry(QtCore.QRect(0, 0, 581, 401))
        self.head_pose_stats.setStyleSheet("background-color:#919191;")
        self.head_pose_stats.setText("")
        self.head_pose_stats.setObjectName("head_pose_stats")
        self.tab_widget.addTab(self.tab_3, "")
        self.tab_4.setObjectName("tab_4")
        self.recording_stats.setGeometry(QtCore.QRect(0, 0, 581, 401))
        self.recording_stats.setStyleSheet("background-color:#919191;")
        self.recording_stats.setText("")
        self.recording_stats.setObjectName("recording_stats")
        self.tab_widget.addTab(self.tab_4, "")
        self.splitter_2.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.check_box.setGeometry(QtCore.QRect(300, 580, 361, 20))
        self.check_box.setObjectName("check_box")
        self.top_label.setGeometry(QtCore.QRect(0, 0, 751, 71))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setUnderline(True)
        self.top_label.setFont(font)
        self.top_label.setStyleSheet("background-color:#758e88;")
        self.top_label.setAlignment(QtCore.Qt.AlignCenter)
        self.top_label.setObjectName("top_label")
        self.tool_button.setGeometry(QtCore.QRect(700, 70, 51, 31))
        self.tool_button.setStyleSheet("border-bottom-left-radius: 10px; "
                                       "background-color:#98b7b0; "
                                       "color:black; "
                                       "height:40;")
        self.tool_button.setObjectName("tool_button")
        self.top_label2.setGeometry(QtCore.QRect(0, 70, 51, 31))
        self.top_label2.setStyleSheet("border-bottom-right-radius: 10px; "
                                      "background-color:#98b7b0; "
                                      "color:black; "
                                      "height:40;")
        self.top_label2.setText("")
        self.top_label2.setObjectName("top_label2")
        self.splitter.setGeometry(QtCore.QRect(80, 530, 581, 40))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.record_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.record_btn.setStyleSheet("border-radius: 10px; "
                                      "background-color:white; "
                                      "color:black; "
                                      "height:40; "
                                      "border:1px solid #9e9e9e;")
        self.record_btn.setCheckable(False)
        self.record_btn.setAutoDefault(False)
        self.record_btn.setFlat(False)
        self.record_btn.setObjectName("record_btn")
        main_window.setCentralWidget(self.centralwidget)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 751, 26))
        self.menubar.setObjectName("menubar")
        main_window.setMenuBar(self.menubar)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.re_translate_ui(main_window)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(main_window)

        # set trigger for the record button
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self.run)

        # set trigger for the help button
        self.tool_button.clicked.connect(self.help_prompt)


    def re_translate_ui(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Final Year Project 2018/19"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab),
                                   _translate("main_window",
                                              "Live Camera"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_2),
                                   _translate("main_window",
                                              "Emotion Stats"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_3),
                                   _translate("main_window",
                                              "Head Pose Stats"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_4),
                                   _translate("main_window",
                                              "Screen Recording"))
        self.check_box.setText(_translate("main_window", "Keep Window Open"))
        self.top_label.setText(_translate("main_window", "Facial Analysis For Task Engagement"))
        self.tool_button.setText(_translate("main_window", "Help"))
        self.record_btn.setText(_translate("main_window", "Record"))
        self.record_btn.setShortcut(_translate("main_window", "Space"))


    @staticmethod
    def get_candidate_id():
        # run loop until the next candidate ID is found in the saves directory
        find_next_candidate = 0

        while os.path.exists(os.path.join(os.path.expanduser('~'),
                                          'Downloads',
                                          'FYP_Data',
                                          'csv-candidate%s.csv') % find_next_candidate):
            find_next_candidate = find_next_candidate + 1

        return find_next_candidate


    @staticmethod
    def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        _x = rect.left()
        _y = rect.top()
        _w = rect.right() - _x
        _h = rect.bottom() - _y

        # return a tuple of (x, y, w, h)
        return _x, _y, _w, _h


    @staticmethod
    def make_np_from_shape(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords


    @staticmethod
    def help_prompt():
        # display app information popup on click
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()


    def get_euler_angle(self, shape, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # get the shape of the image
        size = gray.shape

        # find the landmark image points
        image_points = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                   shape[39], shape[42], shape[45], shape[31], shape[33],
                                   shape[35], shape[48], shape[54], shape[57], shape[8]])

        # calculate the camera matrix and declare coefficients (we assume no lens distortion)
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        # calculate the Rigid Body Transform vectors
        _, rotation_vec, translation_vec = cv2.solvePnP(self.model_points,
                                                        image_points,
                                                        camera_matrix,
                                                        dist_coeffs)

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return euler_angle


    def save_data(self):
        # calculate the Frames per Second
        hrs_mins_secs = '%H:%M:%S'
        diff = dt.datetime.strptime(self.endtime, hrs_mins_secs) - dt.datetime.strptime(self.starttime, hrs_mins_secs)
        secdiff = diff.seconds
        frames_per_second = round(self.frame_counter/secdiff)

        # splice both arrays in increments of the FPS - what was the user doing at that second
        emotion_array = self.emotion_array[0::int(frames_per_second)]
        engaged_array = self.engaged_array[0::int(frames_per_second)]

        # list the items for the x-axis (Capture Time)
        for i in range(0, int(secdiff)):
            self.capture_time_1.append(i)

        # combine our list of emotions with the list of pre-defined emotions
        # (Keeps y-axis list consistent)
        self.appended_emotion_array = self.appended_emotion_array + emotion_array

        # error collection
        for i in range(0, int(secdiff)):
            if len(self.capture_time_1) < len(self.appended_emotion_array):
                self.appended_emotion_array.pop(len(self.appended_emotion_array)-1)
            elif len(self.capture_time_1) > len(self.appended_emotion_array):
                self.appended_emotion_array.append("Neutral")
            else:
                break

        # draw the charts
        # draw scatter plot
        plt.figure(figsize=(20, 10))
        plt.subplot2grid((3, 3), (0, 0), colspan=3)
        plt.axis([0, int(secdiff), 0, 10])
        plt.xlabel('Capture Time (secs)')
        plt.ylabel('Emotions')
        plt.plot(self.capture_time_1,
                 self.appended_emotion_array,
                 color='c')
        plt.scatter(self.capture_time_1,
                    self.appended_emotion_array,
                    color='c',
                    marker='o',
                    s=20)

        # list the items for the x-axis (Capture Time)
        for i in range(0, int(secdiff)):
            self.capture_time_2.append(i)

        # combine our engagement list with the list of pre-defined engagements
        # (Keeps y-axis list consistent)
        self.appended_engagement_array = self.appended_engagement_array + engaged_array

        # error collection
        for i in range(0, int(secdiff)):
            if len(self.capture_time_2) < len(self.appended_engagement_array):
                self.appended_engagement_array.pop(len(self.appended_engagement_array)-1)
            elif len(self.capture_time_2) > len(self.appended_engagement_array):
                self.appended_engagement_array.append("Engaged")
            else:
                break

        # draw the charts
        # draw scatter plot
        plt.subplot2grid((3, 3), (1, 0), colspan=3)
        plt.axis([0, int(secdiff), 0, 3])
        plt.xlabel('Capture Time (secs)')
        plt.ylabel('Engagement')
        plt.plot(self.capture_time_2,
                 self.appended_engagement_array,
                 color='c')
        plt.scatter(self.capture_time_2,
                    self.appended_engagement_array,
                    color='c',
                    marker='o',
                    s=20)

        # data collection for pie chart and bar chart
        x = ['Engaged', 'Not Engaged']
        y = [self.is_engaged, self.is_not_engaged]

        x_bar = ['Fear', 'Disgust', 'Angry', 'Sad', 'Neutral', 'Happy', 'Surprised']
        y_bar = [self.feeling_fear,
                 self.feeling_disgust,
                 self.feeling_angry,
                 self.feeling_sad,
                 self.feeling_neutral,
                 self.feeling_happy,
                 self.feeling_surprised]

        x_pie = []
        y_pie = []

        for i in range(0, len(y_bar)):
            if not y_bar[i] == 0:
                y_pie.append(y_bar[i])
                x_pie.append(x_bar[i])

        #draw pie chart for emotion
        plt.subplot2grid((3, 3), (2, 0), colspan=1)
        plt.pie(y_pie, labels=x_pie, shadow=True, autopct='%1.1f%%')

        #draw bar chart for expressions
        plt.subplot2grid((3, 3), (2, 1), colspan=1)
        plt.bar(x_bar, y_bar)
        plt.xlabel('Emotions')
        plt.ylabel('Value (Frames)')

        #draw pie chart for engagement
        plt.subplot2grid((3, 3), (2, 2), colspan=1)
        plt.pie(y, labels=x, shadow=True, autopct='%1.1f%%')

        # save data
        plt.savefig(os.path.join(os.path.expanduser('~'),
                                 'Downloads',
                                 'FYP_Data',
                                 'chart-candidate%s-results.png') % self.next_candidate_id)

        print("Recording Length (secs): {}".format(secdiff))
        print("frames_per_second): {}".format(frames_per_second))
        print("Engagement Collected: {}".format(len(engaged_array)))
        print("Emotions Captured: {}".format(len(emotion_array)))

        if secdiff > len(emotion_array):
            for i in range(len(emotion_array)-1, secdiff+1):
                emotion_array.append("Not Measured")

        if secdiff > len(engaged_array):
            for i in range(len(engaged_array)-1, secdiff+1):
                engaged_array.append("Not Measured")

        with open(os.path.join(os.path.expanduser('~'),
                               'Downloads',
                               'FYP_Data',
                               'csv-candidate%s.csv') % self.next_candidate_id,
                  'w', newline="") as df:
            fieldnames = ['Current Time (secs)', 'Emotion', 'Head Pose']
            thewriter = csv.DictWriter(df, fieldnames=fieldnames)

            thewriter.writeheader()
            for i in range(0, secdiff):
                thewriter.writerow({'Current Time (secs)' : i,
                                    'Emotion' : emotion_array[i],
                                    'Head Pose' : engaged_array[i]})

        QMessageBox.about(main_window,
                          "Info",
                          "Your Data Has Been Saved To:\n{}".format(
                              os.path.join(os.path.expanduser('~'),
                                           'Downloads',
                                           'final_year_project_data')))


    def run(self, status):
        if status:
            # set new styles to the record button to show that it is recording
            self.record_btn.setText('Stop')
            self.record_btn.setStyleSheet("border-radius: 10px;"
                                          "background-color:red;"
                                          "color:white;"
                                          "height:40;")

            # get the new ID number for the next candidate
            self.next_candidate_id = self.get_candidate_id()

            # initialise the face detection
            self.face_enabled = True

            # register the recordings start time
            self.starttime = dt.datetime.now().strftime("%H:%M:%S")

            # initialize the frame counter and the arrays for data collection
            self.frame_counter = 0
            self.engaged_array = []
            self.emotion_array = []
            self.capture_time_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.capture_time_2 = [0, 0, 0, 0]
            self.appended_emotion_array = ['',
                                           'Absent',
                                           'Angry',
                                           'Sad',
                                           'Neutral',
                                           'Happy',
                                           'Surprise',
                                           'Fear',
                                           'Disgust',
                                           '']
            self.appended_engagement_array = ['Not Measured',
                                              'Engaged',
                                              'Not Engaged',
                                              '']
            self.last_emotion_tracked = "Neutral"

            # initialise the screen recording
            self.sct = mss.mss()
            self.bbox = (100, 10, 1680, 1050)

            # control variables for the recording
            self.set_fps_to_record = True
            self.grab_frame_two = 0

            # check whether to hide the UI to the user or not (for data privacy)
            if not self.check_box.isChecked():
                self.tab_widget.setTabEnabled(0, False)
                self.tab_widget.setCurrentIndex(3)
        else:
            self.stop("_")


    def stop(self, _):
        # register end time and stop face recording
        self.endtime = dt.datetime.now().strftime("%H:%M:%S")
        self.face_enabled = False

        # reset the styles of the record button to show that it is no longer recording
        self.record_btn.setText('Record') # ◉
        self.record_btn.setStyleSheet("border-radius: 10px; "
                                      "background-color:white; "
                                      "color:black; "
                                      "height:40;")

        # clear the canvas'
        self.emotion_stats.clear()
        self.head_pose_stats.clear()
        self.recording_stats.clear()

        # show the main window at all times upon recording termination
        self.tab_widget.setTabEnabled(0, True)
        self.tab_widget.setCurrentIndex(0)

        # pass to process and save data
        self.save_data()


    def update_frame(self):
        # read in a frame and flip the image to match the users motion
        ret, image = self.cap.read()
        image = cv2.flip(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check if detection can occur
        if self.face_enabled:
            last_time = time.time()
            # run function to retrieve an updated frame with all data marked off,
            # then sent it to be displayed
            detected_image, engagement_canvas, emotion_canvas = self.detect_face(image)
            self.frame_counter = self.frame_counter+1

            # Get raw pixels from the screen, save it to a Numpy array
            screenImg = np.array(self.sct.grab(self.bbox))
            screenImg = np.flip(screenImg[:, :, :3], 2)
            screenFrame = cv2.cvtColor(screenImg, cv2.COLOR_BGR2RGB)

            self.display_image(detected_image, "image", 1)
            self.display_image(engagement_canvas, "orientation", 1)
            self.display_image(emotion_canvas, "expression", 1)
            self.display_image(screenFrame, "record", 1)

            # set up recorder for saving frames
            video_fps = (1 / (time.time() - last_time))
            if self.set_fps_to_record == True and self.grab_frame_two == 1:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                self.out = cv2.VideoWriter(os.path.join(
                    os.path.expanduser('~'),
                    'Downloads',
                    'FYP_Data',
                    'screen_recording-candidate%s.avi')% self.next_candidate_id,
                                           fourcc,
                                           video_fps,
                                           (1580, 1040))
                self.set_fps_to_record = False

            # first frame captured can stutter for a brief moment.
            # So we set it to take the second frame FPS value (more representative)
            if not self.grab_frame_two == 1:
                self.grab_frame_two += 1

            if not self.set_fps_to_record:
                self.out.write(screenFrame)

        else:
            self.display_image(image, "image", 1)
            if self.temp:
                self.help_prompt()
                self.temp = False


    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        self.rects = self.detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        if len(self.rects) > 0:
            text = "{} face(s) found".format(len(self.rects))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            self.emotion_array.append("Absent")
            self.engaged_array.append("Not Engaged")
            #self.frame_counter = self.frame_counter+1

        # loop over the face detections
        for self.rect in self.rects:
            (bX, bY, bW, bH) = self.rect_to_bb(self.rect)
            self.start_x = bX - 20
            self.start_y = bY - 50
            self.end_x = bX + bW + 50
            self.end_y = bY + bH + 20

            frame, self.emotion_canvas = self.detect_expression(frame)
            frame, self.engagement_canvas = self.detect_orientation(frame)

        return frame, self.engagement_canvas, self.emotion_canvas


    def detect_orientation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # determine the facial landmarks for the face region,
        # then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, self.rect)
        shape = self.make_np_from_shape(shape)

        euler_angle = self.get_euler_angle(shape, frame)

        # loop over the (x, y)-coordinates for the facial landmarks and draw each of them
        for (i, (x, y)) in enumerate(shape):

            # draw key landmark points in green
            if (i == 33) or (i == 8) or (i == 36) or (i == 45) or (i == 48) or (i == 54):
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    str(i + 1),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1)

            # draw other landmarks in red
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(i + 1),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1)

        # check if user is engaged
        if(euler_angle[0, 0] > 13) \
                or (euler_angle[0, 0] < -13) \
                or (euler_angle[1, 0] > 13) \
                or(euler_angle[1, 0] < -13):
            self.engaged_array.append("Not Engaged")
            self.is_not_engaged += 1
        else:
            self.engaged_array.append("Engaged")
            self.is_engaged += 1

        # grab head coords for yaw, pitch, and roll
        self.headpose_canvas = np.zeros((270, 300), dtype="uint8")
        pos_prediction = [euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]]

        # print x,y,z angles on screen
        for (i, (POS_Labels, prob)) in enumerate(zip(self.pos, pos_prediction)):
            # construct the label text
            text = "{}: {:.2f}%".format(POS_Labels, prob)
            cv2.putText(
                self.headpose_canvas,
                text,
                (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2)

        return frame, self.headpose_canvas


    def detect_expression(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # locate the face and resize
        gray_face = gray[self.start_y:self.end_y, self.start_x:self.end_x]
        self.emotion_canvas = np.zeros((270, 300, 3), dtype="uint8")
        try:
            roi = cv2.resize(gray_face, (64, 64))
        except:
            return frame, self.emotion_canvas

        # pass region of interest into a numpy sequence for working with our deep learning model
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction of the users emotion using the model
        emotion_prediction = self.emotion_classifier.predict(roi)[0]
        print(emotion_prediction)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)

        # create the expression canvas and content
        for (i, (emotion, prob)) in enumerate(zip(self.emotions, emotion_prediction)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(
                self.emotion_canvas,
                (1, (i * 35) + 5),
                (w, (i * 35) + 35),
                (0, 0, 255),
                -1)
            cv2.putText(
                self.emotion_canvas,
                text,
                (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2)

        emotion_text = self.emotion_labels[emotion_label_arg]
        self.emotion_window.append(emotion_text)

        # make a determination of the present emotion
        if len(self.emotion_window) > 10: # 10=frame window
            self.emotion_window.pop(0)
        try:
            emotion_mode = mode(self.emotion_window)
        except:
            self.emotion_array.append(self.last_emotion_tracked)
            return frame, self.emotion_canvas

        self.last_emotion_tracked = emotion_text
        # collect data based on the emotion expressed by the user
        if emotion_text == 'Angry':
            self.color = emotion_probability * np.asarray((0, 0, 255)) #red
            self.feeling_angry += 1
            self.emotion_array.append('Angry')
        elif emotion_text == 'Sad':
            self.color = emotion_probability * np.asarray((219, 73, 57)) #darkish-blue
            self.feeling_sad += 1
            self.emotion_array.append('Sad')
        elif emotion_text == 'Happy':
            self.color = emotion_probability * np.asarray((34, 226, 217)) #yellow
            self.feeling_happy += 1
            self.emotion_array.append('Happy')
        elif emotion_text == 'Surprise':
            self.color = emotion_probability * np.asarray((255, 255, 0)) #light blue
            self.feeling_surprised += 1
            self.emotion_array.append('Surprise')
        elif emotion_text == 'Neutral':
            self.color = emotion_probability * np.asarray((66, 244, 75)) #green
            self.feeling_neutral += 1
            self.emotion_array.append('Neutral')
        elif emotion_text == 'Fear':
            self.color = emotion_probability * np.asarray((209, 27, 187)) #green
            self.feeling_fear += 1
            self.emotion_array.append('Fear')
        elif emotion_text == "Disgust":
            self.color = emotion_probability * np.asarray((66, 244, 75)) #green
            self.feeling_disgust += 1
            self.emotion_array.append('Disgust')

        # set the colour for use with our bounding box
        self.color = self.color.astype(int)
        self.color = self.color.tolist()

        # display data on screen
        y = self.start_y - 10 if self.start_y - 10 > 10 else self.start_y + 10
        cv2.rectangle(
            frame,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            self.color,
            2)
        cv2.putText(
            frame,
            emotion_mode,
            (self.start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.color,
            2, cv2.LINE_AA)
        return frame, self.emotion_canvas


    def display_image(self, img, types, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        if window == 1:
            if types == "image":
                self.display_user.setPixmap(QPixmap.fromImage(out_image))
                self.display_user.setScaledContents(True)
            elif types == "orientation":
                self.head_pose_stats.setPixmap(QPixmap.fromImage(out_image))
                self.head_pose_stats.setScaledContents(True)
            elif types == "record":
                self.recording_stats.setPixmap(QPixmap.fromImage(out_image))
                self.recording_stats.setScaledContents(True)
            else:
                self.emotion_stats.setPixmap(QPixmap.fromImage(out_image))
                self.emotion_stats.setScaledContents(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setup_ui(main_window)
    main_window.show()
    sys.exit(app.exec_())
