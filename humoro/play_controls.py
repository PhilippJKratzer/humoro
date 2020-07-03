#!/usr/bin/python2
from __future__ import print_function

import numpy as np
from humoro.segment_manager import SegmentManager
from PyQt5 import QtWidgets, QtGui, QtCore
import time


class LegendRow(object):

    def __init__(self):
        self.items = []


class LegendItem(object):

    def __init__(self, color, label, width):
        self.color = color
        self.label = label
        self.width = width


class LegendWidget(QtWidgets.QWidget):

    def __init__(self, segment_manager):
        super(LegendWidget, self).__init__()
        self._segman = segment_manager
        self._size = 24.0
        self._layout = []
        self._update_layout()

    def resizeEvent(self, e):
        self._update_layout()

    def paintEvent(self, e):
        p = QtGui.QPainter()
        p.begin(self)

        size = self.size()
        width = size.width()

        size = int(self._size * self.logicalDpiX() / 96.0)
        space = int(4.0 * self.logicalDpiX() / 96.0)

        pos_y = 0
        for row in self._layout:
            pos_x = 0

            for item in row.items:
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(item.color[0], item.color[1], item.color[2]))
                p.drawRect(pos_x, pos_y + space, size - 2 * space, size - 2 * space)

                p.setPen(QtGui.QColor(0, 0, 0))
                p.drawText(QtCore.QRect(pos_x + size, pos_y, width - pos_x, size), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, item.label)
                pos_x += item.width + 4 * space

            pos_y += size + space
        p.end()

    def _update_layout(self):
        fm = QtGui.QFontMetrics(QtGui.QFont())

        size = self.size()
        width = size.width()

        size = int(self._size * self.logicalDpiX() / 96.0)
        space = int(4.0 * self.logicalDpiX() / 96.0)

        items = []
        for l in self._segman.labels:
            color = self._segman.get_color_for_label(l)
            item_width = size + fm.width(l)
            item = LegendItem(color, l, item_width)
            items.append(item)

        self._layout = []
        data = {'row': LegendRow(), 'row_width': 0}

        def append_row(item):
            data['row'].items.append(item)
            data['row_width'] += item.width + 4 * space
            if data['row_width'] >= width:
                next_row()

        def next_row():
            self._layout.append(data['row'])
            data['row'] = LegendRow()
            data['row_width'] = 0

        for item in items:
            if data['row_width'] == 0:
                append_row(item)
            elif data['row_width'] + item.width <= width:
                append_row(item)
            else:
                next_row()
                append_row(item)

        if len(data['row'].items) > 0:
            self._layout.append(data['row'])

        row_count = len(self._layout)
        self.setMinimumSize(1, row_count * size + (row_count - 1) * space)


class SegmentWidget(QtWidgets.QWidget):

    timeChanged = QtCore.pyqtSignal(float)

    def __init__(self, segment_manager, fps):
        self.fps = fps
        super(SegmentWidget, self).__init__()
        self._height_scale = 115.0
        self.setMinimumSize(1, self._height_scale * self.logicalDpiX() / 96.0)
        self._segment_manager = segment_manager
        self._start = segment_manager.segments[0].start
        self._end = segment_manager.segments[-1].end
        self._time = self._start
        self._mouse_offset = 0
        self._overview_width = -1
        self._overview_height = -1
        self._overview_valid = False
        self._overview = None

    @QtCore.pyqtProperty(float)
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.repaint()

    def force_repaint(self):
        self._overview_valid = False
        self.repaint()

    def _handle_mouse_event(self, e):
        size = self.size()
        width = size.width()

        pos = max(0, min(e.x() + self._mouse_offset, width - 1))

        length = float(self._end - self._start)
        self._time = (float(pos) / (width - 1)) * length + self._start
        self.timeChanged.emit(self._time)

    def mousePressEvent(self, e):
        size = self.size()
        width = size.width()

        pos = self._get_pos(self._time, self._start, self._end, width)

        self._mouse_offset = pos - e.x()
        if abs(self._mouse_offset) > 10:
            self._mouse_offset = 0
        self._handle_mouse_event(e)

    def mouseReleaseEvent(self, e):
        # self._handle_mouse_event(e)
        pass

    def mouseMoveEvent(self, e):
        self._handle_mouse_event(e)

    def paintEvent(self, e):
        p = QtGui.QPainter()
        p.begin(self)

        # start
        size = self.size()
        width = size.width()
        height = size.height()

        # draw segments
        height_segments = int(height * 30.0 / self._height_scale)
        pos_segments = 0

        length = min(self._end - self._start, 5000)
        ratio = (self._time - self._start) / (self._end - self._start)
        start = self._time - ratio * length
        end = self._time + (1 - ratio) * length
        self._draw_segments(p, width, height_segments, pos_segments, start, end, 0.5)

        height_segments = int(height * 30.0 / self._height_scale)
        pos_segments = int(height * 40.0 / self._height_scale)

        overview_width = width
        overview_height = height_segments
        size_changed = (self._overview_width != overview_width or self._overview_height != overview_height)
        if self._overview is None or not self._overview_valid or size_changed:
            if size_changed:
                self._overview = QtGui.QPixmap(overview_width, overview_height)
            p_overview = QtGui.QPainter()
            p_overview.begin(self._overview)
            self._draw_segments(p_overview, overview_width, overview_height, 0, self._start, self._end, 0.25)
            p_overview.end()
            self._overview_width = overview_width
            self._overview_height = overview_height
            self._overview_valid = True
        p.drawPixmap(0, pos_segments, self._overview)

        # draw progress

        height_progress = int(height * 10.0 / self._height_scale)
        pos_progress = int(height * 80.0 / self._height_scale)

        pos = self._get_pos(self._time, self._start, self._end, width)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(100, 180, 255))
        p.drawRect(0, pos_progress, pos + 1, height_progress)
        p.setBrush(QtGui.QColor(200, 200, 200))
        p.drawRect(pos + 1, pos_progress, width - pos, height_progress)

        p.setPen(QtGui.QColor(0, 0, 0, 70))
        p.drawLine(pos, 0, pos, pos_progress + height_progress - 1)

        # draw time

        p.setPen(QtGui.QColor(0, 0, 0))

        time_start = 0
        text_start = self._format_time(time_start)
        p.drawText(QtCore.QRect(0, 0, width, height), QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft, text_start)

        time_end = self._end - self._start
        text_end = self._format_time(time_end)
        p.drawText(QtCore.QRect(0, 0, width, height), QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight, text_end)

        time_pos = self._time - self._start
        text_pos = self._format_time(time_pos)
        p.drawText(QtCore.QRect(0, 0, width, height), QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, text_pos)

        # end
        p.end()

    def _get_pos(self, time, start, end, width):
        length = float(end - start)
        pos = int((width - 1) * (time - start) / length + 0.5)
        return pos

    def _draw_segments(self, p, width, height, pos_y, start, end, line_opacity):
        segman = self._segment_manager
        index_start = segman.get_at_index(start)

        # boxes

        p.setPen(QtCore.Qt.NoPen)

        i = index_start
        while i < len(segman.segments):
            s = segman.segments[i]
            pos_start = self._get_pos(s.start, start, end, width)
            pos_end = self._get_pos(s.end, start, end, width)
            if pos_start > width - 1:
                break

            color = segman.get_color_for_label(s.label)
            p.setBrush(QtGui.QColor(color[0], color[1], color[2]))
            pos_start = max(0, pos_start)
            pos_end = min(width - 1, pos_end)
            p.drawRect(pos_start, pos_y, pos_end, height)

            i += 1

        # lines

        p.setPen(QtGui.QColor(0, 0, 0, int(255 * line_opacity + 0.5)))

        pos = self._get_pos(self._start, start, end, width)
        if pos >= 0 and pos <= width - 1:
            p.drawLine(pos, pos_y, pos, pos_y + height - 1)

        i = index_start
        while i < len(segman.segments):
            s = segman.segments[i]
            pos = self._get_pos(s.end, start, end, width)
            if pos > width - 1:
                break

            p.drawLine(pos, pos_y, pos, pos_y + height - 1)

            i += 1

    def _format_time(self, time):
        time_msec = int(time * (100. / self.fps))
        time_sec = time / self.fps
        time_min = time_sec / 60
        return "%d:%02d.%02d" % (time_min, time_sec % 60, time_msec % 1000 / 10)


class Window(QtWidgets.QMainWindow):

    def __init__(self, path_segments=None, playback_func=None, time_start=None, time_end=None, fps=120, parent=None):
        super(Window, self).__init__(parent)

        self.fps = fps
        self._playback_func = playback_func
        self._time_start = time_start
        self._time_end = time_end
        self._time = self._time_start
        self._segment_manager = SegmentManager(path_segments, self._time_start, self._time_end)

        widget_vbox = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(10.0 * self.logicalDpiY() / 96.0)
        widget_vbox.setLayout(vbox)
        self.setCentralWidget(widget_vbox)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setSpacing(8.0 * self.logicalDpiX() / 96.0)
        vbox.addLayout(hbox)

        self._button_play = QtWidgets.QPushButton("play")
        self._button_play.setCheckable(True)
        self._button_play.clicked.connect(self.on_button_play_clicked)
        hbox.addWidget(self._button_play)

        self._button_split = QtWidgets.QPushButton("split")
        self._button_split.clicked.connect(self.on_button_split_clicked)
        hbox.addWidget(self._button_split)

        self._button_merge = QtWidgets.QPushButton("merge")
        self._button_merge.clicked.connect(self.on_button_merge_clicked)
        hbox.addWidget(self._button_merge)

        self._combobox_label = QtWidgets.QComboBox()
        print (self._segment_manager.labels)
        self._combobox_label.insertItems(0, self._segment_manager.labels)
        self._combobox_label.activated.connect(self.on_combobox_label_activated)
        hbox.addWidget(self._combobox_label)

        self._segment_widget = SegmentWidget(self._segment_manager, self.fps)
        self._segment_widget.timeChanged.connect(self.on_segment_widget_timeChanged)
        vbox.addWidget(self._segment_widget)

        legend_widget = LegendWidget(self._segment_manager)
        vbox.addWidget(legend_widget)

        self._index_label = QtWidgets.QLabel("Mocap_index: ")
        vbox.addWidget(self._index_label)

        hbox_intent = QtWidgets.QHBoxLayout()
        hbox_intent.setSpacing(8.0 * self.logicalDpiX() / 96.0)
        vbox.addLayout(hbox_intent)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.on_timer_timeout)
        timer.start(16)

        self.setGeometry(100, 100, 600.0 * self.logicalDpiX() / 96.0, 1)
        self.setFixedHeight(self.sizeHint().height())

        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()

        self._print_stats()

    def paintEvent(self, event):
        self.setFixedHeight(self.sizeHint().height())

    def closeEvent(self, event):
        self._segment_manager.close()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.on_button_play_clicked()
        elif key == QtCore.Qt.Key_S:
            self.on_button_split_clicked()
        elif key == QtCore.Qt.Key_M:
            self.on_button_merge_clicked()
        elif key == QtCore.Qt.Key_L:
            self._combobox_label.showPopup()
        elif key == QtCore.Qt.Key_Up:
            self._change_time(-10)
        elif key == QtCore.Qt.Key_Down:
            self._change_time(10)
        elif key == QtCore.Qt.Key_Left:
            self._change_time(-1)
        elif key == QtCore.Qt.Key_Right:
            self._change_time(1)
        elif key == QtCore.Qt.Key_PageUp:
            self._change_time(-100)
        elif key == QtCore.Qt.Key_PageDown:
            self._change_time(100)
        elif key == QtCore.Qt.Key_Home:
            self._change_time_abs(self._time_start)
        elif key == QtCore.Qt.Key_End:
            self._change_time_abs(self._time_end)

    def _change_time(self, offset):
        time = self._time
        time += offset
        self._change_time_abs(time)

    def _change_time_abs(self, time):
        if time < self._time_start:
            time = self._time_start
        elif time > self._time_end:
            time = self._time_end
        self._time = time
        self._update_time()

    def _update_time(self):
        self._playback_func(self._time)
        self._segment_widget.time = self._time
        self._index_label.setText("Frame: " + str(int(self._time)))

    def _update_label(self):
        time = self._time
        s = self._segment_manager.get_at(time)
        index = self._segment_manager.labels.index(s.label)
        self._combobox_label.setCurrentIndex(index)

    def on_button_split_clicked(self):
        time = self._time
        self._segment_manager.split_at(time)
        self._segment_widget.force_repaint()

    def on_button_merge_clicked(self):
        time = self._time
        self._segment_manager.merge_at(time)
        self._segment_widget.force_repaint()

    def on_combobox_label_activated(self):
        time = self._time
        s = self._segment_manager.get_at(time)
        s.label = self._combobox_label.currentText()
        self._segment_widget.force_repaint()

    def on_button_play_clicked(self):
        self._prev_time = time.time()

    def on_segment_widget_timeChanged(self, value):
        self._time = value
        self._update_label()

    def on_timer_timeout(self):
        if self._button_play.isChecked():
            passed_frames = (time.time() - self._prev_time) * float(self.fps)
            self._prev_time = time.time()
            self._change_time(passed_frames)

        self._update_time()
        self._update_label()

    def _print_stats(self):
        time_total = 0
        labels = {}
        for l in self._segment_manager.labels:
            labels[l] = []

        segments = self._segment_manager.segments
        for s in segments:
            length = (s.end - s.start) / 120.
            time_total += length
            labels[s.label].append(length)

        for l in labels:
            items = labels[l]

            print ("")
            print ("%s:" % l)

            count = len(items)
            count_fraction = float(count) / len(segments)
            print ("  count: %d (%.3f)" % (count, count_fraction))

            time = sum(items)
            time_fraction = float(time) / time_total
            print ("  time sum: %.1f (%.3f)" % (time, time_fraction))

            time_avg = 0 if len(items) == 0 else float(time) / len(items)
            time_min = 0 if len(items) == 0 else min(items)
            time_max = 0 if len(items) == 0 else max(items)
            time_std = 0 if len(items) == 0 else np.std(items)
            print ("  time avg (std dev) [min/max]: %.1f (%.1f) [%.1f/%.1f]" % (time_avg, time_std, time_min, time_max))


def startwindow(path_segments=None, playback_func=None, time_start=None, time_end=None, fps=None):
    app = QtWidgets.QApplication(["Playback Controls"])
    window = Window(path_segments=path_segments, playback_func=playback_func, time_start=time_start, time_end=time_end, fps=fps)
    window.show()
    app.exec_()


if __name__ == "__main__":
    app = startwindow()
