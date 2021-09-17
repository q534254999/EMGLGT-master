# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect,
                          QSize, QTime, QUrl, Qt, QEvent, QRectF, pyqtSignal)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence,
                         QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient, QMouseEvent)
from PyQt5.QtWidgets import *
import sys
import qtawesome
from settingDialog import Ui_Dialog


class SettingDialog(QtWidgets.QDialog):
    _comlist = pyqtSignal(str, str, str)
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.ui.listWidget.setStyleSheet('''
            *{background-color:#303030;}
            QLabel{
            color:#ffffff;
            border:none;
            font-weight:600;
            font-size:16px;
            font-family:'微软雅黑';
             }
            QPushButton{

            border:none;
            font-weight:600;
            font-size:16px; 
            font-family:'微软雅黑';
             }
            ''')
        # self.ui.setting_title.setStyleSheet('''
        #     QPushButton{ text-align:left;padding-left:30px;color:#ffffff;font-size:16px;}
        #     ''')
        self.m_flag = False
        self.btn = self.ui.pushButton
        self.btn.clicked.connect(self.slot)

    def slot(self):
        port1 = self.ui.comboBox.currentText()
        port2 = self.ui.comboBox_2.currentText()
        port3 = self.ui.comboBox_3.currentText()
        self._comlist.emit(port1, port2, port3)
        # print(self.ui.comboBox.currentText())
        # print(self.ui.comboBox_2.currentText())
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self.m_flag:
            self.move(event.globalPos() - self.m_Position)  # 更改窗口位置
            event.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = SettingDialog()
    gui.show()
    sys.exit(app.exec_())
