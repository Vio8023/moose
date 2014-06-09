# This file is part of MOOSE simulator: http://moose.ncbs.res.in.

# MOOSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MOOSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with MOOSE.  If not, see <http://www.gnu.org/licenses/>.

"""
NeuroML.py is the preferred interface to read NeuroML files.

Instantiate NeuroML class, and thence use method: readNeuroMLFromFile(...) to
load NeuroML from a file:

- The file could contain all required levels 1, 2 and 3 - Network , Morph and
 Channel.

- The file could have only L3 (network) with L1 (channels/synapses) and L2
 (cells) spread over multiple files; these multiple files should be in the same
 directory named as <chan/syn_name>.xml or <cell_name>.xml or
 <cell_name>.morph.xml (essentially as generated by neuroConstruct's export).

- But, if your lower level L1 and L2 xml files are elsewise, use the separate
 Channel, Morph and NetworkML loaders in _moose.neuroml.<...> .

For testing, you can also call this from the command line with a neuroML file
as argument.

CHANGE LOG:

 Description: class NeuroML for loading NeuroML from single file into MOOSE

 Version 1.0 by Aditya Gilra, NCBS, Bangalore, India, 2011 for serial MOOSE

 Version 1.5 by Niraj Dudani, NCBS, Bangalore, India, 2012, ported to parallel

 MOOSE Version 1.6 by Aditya Gilra, NCBS, Bangalore, India, 2012, further
 changes for parallel MOOSE.

 Dilawar Singh; Fixed parsing errors when parsing some standard models.

"""

import sys
import os
from xml.etree import cElementTree as ET

import MorphML
import NetworkML
import ChannelML

from .. import _moose
from .. import print_utils
from .. import moose_config as config

from ..helper import neuroml_utils as mnu

current_version = sys.version_info

if current_version < (2, 6):
    pythonLessThan26 = True
else:
    pythonLessThan26 = False

class NeuroML(object):

    """
    This class parses neuroml models and build _moose-data structures.

    """
    def __init__(self):
        self.lengthUnits = ""
        self.temperature = 25
        self._CELSIUS_default = ""
        self.temperature_default = True
        self.nml_params = None
        self.channelUnits = "Physiological Units"
        _moose.Neutral('/neuroml')
        self.libraryPath = config.libraryPath
        self.cellPath = config.cellPath

    def readNeuroMLFromFile(self, filename, params=dict()):

        """
        For the format of params required to tweak what cells are loaded,
        refer to the doc string of NetworkML.readNetworkMLFromFile().
        Returns (populationDict,projectionDict),
        see doc string of NetworkML.readNetworkML() for details.
        """

        print_utils.dump("STEP"
                , "Loading neuroml file `{0}` ... ".format(filename)
                )
        # creates /library in MOOSE tree; elif present, wraps
        tree = ET.parse(filename)
        root_element = tree.getroot()
        self.modelDir = os.path.dirname(os.path.abspath(filename))
        try:
            self.lengthUnits = root_element.attrib['lengthUnits']
        except KeyError:
            self.lengthUnits = root_element.attrib['length_units']
        except Exception as e:
            print_utils.dump("WARN"
                    , "Failed to get length_unit"
                    , sys.exec_info()[0]
                    )
            raise e

        # gets replaced below if tag for temperature is present
        self.temperature = self._CELSIUS_default

        for mp in root_element.findall('.//{'+mnu.meta_ns+'}property'):
            tagname = mp.attrib['tag']
            if 'temperature' in tagname:
                self.temperature = float(mp.attrib['value'])
                self.temperature_default = False
        if self.temperature_default:
            print_debug.dump(
                    "Using default temperature of %s C".format(self.temperature)
                    )
        self.nml_params = {
                'temperature': self.temperature
                , 'model_dir': self.modelDir
                }

        mmlR = MorphML.MorphML(self.nml_params)
        self.cellsDict = {}
        for cells in root_element.findall('.//{'+mnu.neuroml_ns+'}cells'):
            for cell in cells.findall('.//{'+mnu.neuroml_ns+'}cell'):
                cellDict = mmlR.readMorphML(
                        cell
                        , params={}
                        , lengthUnits=self.lengthUnits
                        )
                self.cellsDict.update(cellDict)

        nmlR = NetworkML.NetworkML(self.nml_params)
        populationDict, projectionDict = nmlR.readNetworkML(
                root_element
                , self.cellsDict
                , params=params
                , lengthUnits=self.lengthUnits
                )

        # Loading channels and synapses into MOOSE into neuroml library
        cmlR = ChannelML.ChannelML(self.nml_params)
        return populationDict, projectionDict

    def channelToMoose(self, cmlR, channels):
        for channel in channels.findall('.//{'+mnu.cml_ns+'}channel_type'):
            # ideally I should read in extra params
            # from within the channel_type element and put those in also.
            # Global params should override local ones.
            cmlR.readChannelML(channel, params={}, units=self.channelUnits)
        for synapse in channels.findall('.//{'+mnu.cml_ns+'}synapse_type'):
            cmlR.readSynapseML(synapse, units=self.channelUnits)
        for ionConc in channels.findall('.//{'+mnu.cml_ns+'}ion_concentration'):
            cmlR.readIonConcML(ionConc, units=self.channelUnits)

    def loadNML(self, nmlFile):
        """ Load a model given in NeuroML file """
        return self.readNeuroMLFromFile(nmlFile)
