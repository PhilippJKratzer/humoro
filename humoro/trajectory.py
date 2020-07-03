import numpy as np
import copy


class Trajectory(object):
    """ Interface class for motion trajectories """

    def __init__(self, data=None, description=None, startframe=0, data_fixed={}):
        """ init function

        Keyword arguments:
        data -- 2d array (frameid x state) of trajectory data
        description -- description of the state variables (e.g. names of jointstates)
        startframe -- frame when the trajectory starts (used for playback)
        data_fixed -- map(jointname: value) of state dimensions that stay fixed e.g. scaling parameters
        """
        self.data_fixed = data_fixed
        self.data = data
        self.description = description
        if data is not None:
            self.startframe = startframe

    def storeTrajHDF5(self, filename):
        """ Stores the trajectory in hdf5 file format

        Keyword arguments:
        filename -- path to the file
        """
        import h5py
        h5file = h5py.File(filename, 'w')
        dset = h5file.create_dataset('data', data=self.data)
        dset.attrs['description'] = self.description
        dset.attrs['startframe'] = self.startframe
        keys_fixed, values_fixed = list(self.data_fixed.keys()), list(self.data_fixed.values())
        if len(keys_fixed) > 0:
            dset.attrs['keys_fixed'] = keys_fixed
            dset.attrs['values_fixed'] = values_fixed
        h5file.close()

    def loadTrajHDF5(self, filename):
        """ Loads the trajectory from hdf5 file format

        Keyword arguments:
        filename -- path to the file
        """
        import h5py
        h5file = h5py.File(filename, 'r')
        dset = h5file['data']
        self.data = dset[:]
        self.description = dset.attrs['description']
        if type(self.description[0]) == np.bytes_:
            self.description = [x.decode('utf-8') for x in self.description]

        self.startframe = dset.attrs['startframe']
        if 'keys_fixed' in dset.attrs:
            keys = dset.attrs['keys_fixed']
            if type(keys[0]) == np.bytes_:
                keys = [x.decode('utf-8') for x in keys]
            self.data_fixed = dict(zip(keys, dset.attrs['values_fixed']))
        else:
            self.data_fixed = {}
        h5file.close()

    def append(self, data):
        """ Appends data to the trajectory

        Keyword arguments:
        data -- data to append
        """
        self.data = np.concatenate([self.data, data])
        self.endframe = self.startframe + len(self.data)

    @property
    def startframe(self):
        return self._startframe

    @startframe.setter
    def startframe(self, value):
        """ changes the start frame of the trajectory and adapts the endframe (used for playback)

        Keyword arguments:
        value -- new value of startframe
        """
        self._startframe = value
        self.endframe = value + len(self.data)

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, value):
        """ changes the start frame of the trajectory and adapts the endframe (used for playback)

        Keyword arguments:
        value -- new value of startframe
        """
        self._description = value
        if value is not None:
            self.inv_ind = {}
            for idx, dsc in enumerate(value):
                self.inv_ind[dsc] = idx

    def subTraj(self, startframe, endframe):
        """ Returns a sequence of the trajectory from startframe to endframe

        """
        res = Trajectory(data=copy.deepcopy(self.data[startframe:endframe]), description=copy.deepcopy(self.description), data_fixed=copy.deepcopy(self.data_fixed), startframe=self.startframe + startframe)
        return res

    def getFrameByNames(self, fid, names):
        frame = []
        for n in names:
            if n in self.description:
                frame.append(self.data[fid, self.inv_ind[n]])
            else:
                frame.append(self.data_fixed[n])
        return frame
