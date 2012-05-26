# trbchan.py --- 
# 
# Filename: trbchan.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri May  4 14:55:52 2012 (+0530)
# Version: 
# Last-Updated: Sat May 26 15:48:24 2012 (+0530)
#           By: subha
#     Update #: 36
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Base class for channels in Traub model.
# 
# 

# Change log:
# 
# 2012-05-04 14:55:56 (+0530) subha started porting code from
# channel.py in old moose version to dh_branch.
# 

# Code:

import numpy as np
import moose

class ChannelBase(moose.HHChannel):
    vmin = -120e-3
    vmax = 40e-3
    ndivs = 640
    v_array = np.linspace(vmin, vmax, ndivs+1)
    def __init__(self, path, xpower=1, ypower=0, Ek=0.0):
        if moose.exists(path):
            moose.HHChannel.__init__(self, path)
            return
        moose.HHChannel.__init__(self, path)
        self.Ek = Ek
        if xpower != 0:
            self.Xpower = xpower
            self.xGate = moose.HHGate('%s/gateX' % (path))
            self.xGate.min = ChannelBase.vmin
            self.xGate.max = ChannelBase.vmax
            self.xGate.divs = ChannelBase.ndivs
        if ypower != 0:
            self.Ypower = ypower
            self.yGate = moose.HHGate('%s/gateY' % (path))
            self.yGate.min = ChannelBase.vmin
            self.yGate.max = ChannelBase.vmax
            self.yGate.divs = ChannelBase.ndivs


# 
# trbchan.py ends here