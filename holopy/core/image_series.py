# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""Working with series of images. Lazy loaded using pims so you can
work with large data sets without clobbering your memory

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>

"""

from holopy.core.marray import Image
from pims import ImageSequence
from holopy.core.io import load

import numpy as np

class ImageSeries(ImageSequence):
    def __init__(self, pathname, spacing, optics, frame_time=None, origin=np.zeros(3), metadata={}):
        self.spacing = spacing
        self.optics = optics
        self.frame_time = frame_time
        self.origin = origin
        self.metadata=metadata

        self.pathname = os.path.abspath(pathname)
        if os.path.isdir(pathname):
            warn("Loading ALL files in this directory. To ignore extraneous "
                 "files, use a pattern like 'path/to/images/*.png'",
                 UserWarning)
            directory = pathname
            filenames = os.listdir(directory)
            make_full_path = lambda filename: (
                os.path.abspath(os.path.join(directory, filename)))
            filepaths = list(map(make_full_path, filenames))
        else:
            filepaths = glob.glob(pathname)
        filepaths.sort()  # listdir returns arbitrary order
        self._filepaths = filepaths
        self._count = len(self._filepaths)
        self._first_frame_shape = self.get_frame(0)

        # If there were no matches, this was probably a user typo.
        if self._count == 0:
            raise IOError("No files were found matching that path.")

    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        return self.process_func(load(self._filepaths[j], optics, spacing))
