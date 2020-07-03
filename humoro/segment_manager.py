import h5py
import numpy as np


class Segment(object):

    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label


class SegmentManager(object):

    COLOR_MAP = [[200, 200, 200], [255, 0, 0], [0, 0, 200], [25, 127, 25], [0, 200, 0], [200,0 ,0], [0, 255, 0], [0, 0, 255], [255, 182, 193], [200, 0, 200], [255, 255, 255]]

    def get_color_for_label(self, label):
        if self.labels.index(label) < len(self.COLOR_MAP):
            return self.COLOR_MAP[self.labels.index(label)]
        else:
            return [100, 100, 100]

    def __init__(self, filename, start, end, labels=[], readonly=False):
        self.labels = labels
        self.segments = []
        self._readonly = readonly

        if filename is None:
            self.segments.append(Segment(start, end, "null"))
            self.labels = ["null"]
            self._readonly = True
            return

        self._file = h5py.File(filename, 'a' if not self._readonly else 'r')
        if 'segments' not in self._file:
            self._dt = np.dtype([('start', np.int64), ('end', np.int64), ('label', 'S32')])
            self._dataset = self._file.create_dataset('segments', (0,), maxshape=(None,), dtype=self._dt, chunks=True)
            self.segments.append(Segment(start, end, self.labels[0]))
            self._save()
        else:
            self._dataset = self._file['segments']
            self._load()
            if self.segments[-1].end < end:
                self.segments.append(Segment(self.segments[-1].end, end, "null"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._file is not None:
            self._save()
            self._file.close()
            self._file = None

    def _load(self):
        if "labels" in self._dataset.attrs:
            self.labels = self._dataset.attrs['labels'].tolist()
        for i in range(len(self._dataset)):
            start = self._dataset[i, 'start']
            end = self._dataset[i, 'end']
            label = self._dataset[i, 'label']
            self.segments.append(Segment(start, end, label.decode('utf-8')))

    def _save(self):
        if not self._readonly:
            self._dataset.resize((len(self.segments),))
            self._dataset.attrs._replace = self.labels
            for i, s in enumerate(self.segments):
                self._dataset[i, 'start'] = s.start
                self._dataset[i, 'end'] = s.end
                self._dataset[i, 'label'] = s.label
            self._file.flush()

    def split_at(self, time):
        i = self.get_at_index(time)
        s = self.segments[i]
        if s.start < time and s.end > time:
            s2 = Segment(time, s.end, self.labels[0])
            self.segments.insert(i + 1, s2)
            s.end = time

    def merge_at(self, time):
        if len(self.segments) >= 2:
            i = self.get_at_index(time)
            s = self.segments[i]
            if i == 0 or ((s.end - time) / (s.end - s.start) < 0.5 and i != len(self.segments) - 1):
                s.end = self.segments[i + 1].end
                self.segments.pop(i + 1)
            else:
                s.start = self.segments[i - 1].start
                self.segments.pop(i - 1)

    def get_at_index(self, time):
        if self.segments[0].start >= time:
            return 0
        for i, s in enumerate(self.segments):
            if s.start < time and s.end >= time:
                return i
        return len(self.segments) - 1

    def get_at(self, time):
        return self.segments[self.get_at_index(time)]
