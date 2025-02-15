import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class RecordVideo(QtCore.QObject):
	image_data = QtCore.pyqtSignal(np.ndarray)

	def __init__(self, camera_port=0, parent=None):
		super().__init__(parent)
		self.camera = cv2.VideoCapture(camera_port)

		self.timer = QtCore.QBasicTimer()

	def start_recording(self):
		self.timer.start(0, self)

	def timerEvent(self, event):
		if (event.timerId() != self.timer.timerId()):
			return

		read, data = self.camera.read()
		fdata = cv2.flip(data, 1)
		if read:
			self.image_data.emit(fdata)


class FaceDetectionWidget(QtWidgets.QWidget):
	def __init__(self, haar_cascade_filepath, parent=None):
		super().__init__(parent)
		self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
		self.image = QtGui.QImage()
		self._red = (0, 0, 255)
		self._width = 2
		self._min_size = (30, 30)

	def detect_faces(self, image: np.ndarray):
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image = cv2.equalizeHist(gray_image)

		faces = self.classifier.detectMultiScale(gray_image,scaleFactor=1.3,minNeighbors=4,flags=cv2.CASCADE_SCALE_IMAGE,minSize=self._min_size)

		return faces

	def image_data_slot(self, image_data):
		faces = self.detect_faces(image_data)
		for (x, y, w, h) in faces:
			cv2.rectangle(image_data,
						  (x, y),
						  (x+w, y+h),
						  self._red,
						  self._width)
			cv2.putText(image_data, 'text', (x, y+ 35),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

		self.image = self.get_qimage(image_data)
		if self.image.size() != self.size():
			self.setFixedSize(self.image.size())

		self.update()

	def get_qimage(self, image: np.ndarray):
		height, width, colors = image.shape
		bytesPerLine = 3 * width
		QImage = QtGui.QImage

		image = QImage(image.data,
					   width,
					   height,
					   bytesPerLine,
					   QImage.Format_RGB888)

		image = image.rgbSwapped()
		return image

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawImage(0, 0, self.image)
		self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
	def __init__(self, haarcascade_filepath, parent=None):
		super().__init__(parent)
		fp = haarcascade_filepath
		self.face_detection_widget = FaceDetectionWidget(fp)

		# TODO: set video port
		self.record_video = RecordVideo()

		image_data_slot = self.face_detection_widget.image_data_slot
		self.record_video.image_data.connect(image_data_slot)

		layout = QtWidgets.QVBoxLayout()

		layout.addWidget(self.face_detection_widget)
		self.run_button = QtWidgets.QPushButton('Start')
		layout.addWidget(self.run_button)

		self.run_button.clicked.connect(self.record_video.start_recording)
		self.setLayout(layout)


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	haar_cascade_filepath = cv2.data.haarcascades +'haarcascade_frontalface_default.xml'
	main_window = QtWidgets.QMainWindow()
	main_widget = MainWidget(haar_cascade_filepath)
	main_window.setCentralWidget(main_widget)
	main_window.show()
	sys.exit(app.exec_())