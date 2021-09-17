import sys
from PyQt5.QtCore import *
from PyQt5.Qt import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel


class CirBar(QWidget):
    def __init__(self):
        super(CirBar, self).__init__()
        # 去边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        # 设置窗口背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.persent = 0

    # 定义更新
    def GenUpdate(self, p):
        self.persent = p

    # 定义绘画事件
    def paintEvent(self, event):
        # 360°分成100等分

        rotateAngle = 360 * self.persent / 100
        # 绘制准备工作，启用反锯齿
        painter = QPainter(self)
        # 启用反锯齿，如果本行注释，那么圆的外线有锯齿，不光滑。
        painter.setRenderHints(QPainter.Antialiasing)
        # 设置字体
        font = QFont()
        font.setFamily("Microsoft YaHei")  # 字体种类
        font.setLetterSpacing(QFont.AbsoluteSpacing, 0)  # 间距
        font.setPixelSize(28)  # 像素大小设置，注意不是字号大小。
        painter.setFont(font)

        painter.setOpacity(0.8)   # 透明度
        # 画弦(渐变）
        one = (255 + 255) / 100
        r, g, b = 0, 0, 0
        if self.persent < 50:
            r = one * self.persent
            g = 255
        else:
            g = 255 - ((self.persent - 50) * one)
            r = 255
        r, g, b = int(r), int(g), int(b)
        painter.setBrush(QColor(r, g, b, 255))
        # painter.setBrush(QBrush(QColor("yellow")))
        painter.drawChord(QRectF(15, 6, 96, 96), int((-self.persent * 1.8 + 270) * 16), int(self.persent * 3.6 * 16))
        # 角度渐变(QConicalGradient)
        gradient = QConicalGradient(50, 50, 91)
        # 进度条的画笔颜色
        gradient.setColorAt(1, QColor("blue"))
        self.pen = QPen()
        self.pen.setBrush(gradient)  # 设置画刷渐变效果
        self.pen.setWidth(5)
        self.pen.setCapStyle(Qt.RoundCap)
        painter.setPen(self.pen)        # 250和250是圆点的坐标
        painter.drawArc(QRectF(15, 6, 98, 98), int((90 - 0) * 16), -int(rotateAngle * 16))  # 画圆环

        # 中间画笔的颜色，显示动态百分数的颜色
        painter.setPen(QColor("green"))
        # 画中间动态百分比的文字设置和250和250是圆点的坐标
        painter.drawText(QRectF(15, 6, 98, 98), Qt.AlignCenter, "%d%%" % self.persent)  # 显示进度条当前进度
        self.update()
