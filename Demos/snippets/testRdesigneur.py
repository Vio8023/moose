##################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2015 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
##
## testRdesigneur.py: Builds a spiny compartment and populates it with
## a molecule that diffuses.
##################################################################

import math
import pylab
import numpy
import matplotlib.pyplot as plt
import moose
import sys
sys.path.append( '../util' )
import rdesigneur as rd
from PyQt4 import QtGui
#import moogli_viewer as mv
import moogli

RM = 1.0
RA = 1.0
CM = 0.01
runtime = 10.0
diffConst = 1e-12
dendLen = 100e-6
numDendSegments = 50
segLen = dendLen / numDendSegments
spineSpacing = 2.0e-6
spineSpacingDistrib = 0.0
spineSizeDistrib = 0.2

def makeModel():
    rdes = rd.rdesigneur( useGssa = False, \
                combineSegments = False, \
                cellPortion = "/model/elec/#",  \
                meshLambda = 1e-6, \
                adaptorList = [ \
                    ( 'psd', '.', 'inject', 'Ca', False, 0, 2e-9 ) \
                    ], \
                addSpineList = [ \
                    ( 'spine', '#', \
                    spineSpacing, spineSpacingDistrib, spineSizeDistrib, \
                    0.0, 0.0, numpy.pi, numpy.pi / 2.0 ) \
                    ]
                )
    # Make a 'bare' spine: No synchans, no ion chans, no Ca.
    rdes.addSpineProto( 'spine', RM, RA, CM, \
             synList = (), chanList = (), caTau = 0.0 )
    elec = moose.Neutral( '/tempelec' )
    '''
    ecompt = rdes._buildCompt( elec, 'dend', 100e-6, 2.0e-6, 0, RM, RA, CM )
    ecompt.x0 = 0
    ecompt.x = 0
    ecompt.y0 = 0
    ecompt.y = 0
    ecompt.z0 = 0
    ecompt.z = 100e-6
    '''
    ecompt = []
    for i in range( numDendSegments ):
        ec = rdes._buildCompt( elec, 'dend' + str(i), segLen, 2.0e-6, i * segLen, RM, RA, CM )
        ecompt.append( ec )
        if i > 0:
            moose.connect( ecompt[i-1], 'raxial', ec, 'axial' )
    for i in ecompt:
        i.z0 = i.x0
        i.x0 = 0
        i.z = i.x
        i.x = 0

    chem = moose.Neutral( '/tempchem' )
    for i in ( 'dend', 'spine', 'psd' ):
        compt = moose.CubeMesh( '/tempchem/' + i )
        compt.volume = 1e-18
        ca = moose.Pool( '/tempchem/' + i + '/Ca' )
        ca.concInit = 0.08e-3
        ca.diffConst = 1e-12
    rdes.buildFromMemory( '/tempelec', '/tempchem' )

def addPlot( objpath, field, plot, tick ):
    if moose.exists( objpath ):
        tab = moose.Table( '/graphs/' + plot )
        obj = moose.element( objpath )
        moose.connect( tab, 'requestOut', obj, field )
        tab.tick = tick
        return tab 
    else:
        print "failed in addPlot(", objpath, field, plot, tick, ")" 
        return 0

def plotVm( plot, name ):
    wc = moose.wildcardFind( '/model/elec/' + name + '#' )
    Vm = []
    xpos = []
    for i in wc:
        Vm.append( i.Vm )
        xpos.append( i.z0 )
    if len( wc ) > 0:
        plot.plot( xpos, Vm, label = name + 'Vm' )
    

def main():
    """
    This illustrates the use of rdesigneur to build a simple dendrite with
    spines, and to confirm that the chemical contents of the spines align
    with the electrical. Just a single molecule Ca is involved.
    It diffuses and we plot the distribution.
    It causes 'inject' of the relevant compartment to rise.
    """
    makeModel()

    # Create the output tables
    graphs = moose.Neutral( '/graphs' )
    #dendVm = addPlot( 'model/elec/dend', 'getVm', 'dendVm', 8 )
    addPlot( 'model/chem/dend/Ca[0]', 'getConc', 'dCa0', 18 )
    addPlot( 'model/chem/dend/Ca[25]', 'getConc', 'dCa25', 18 )
    addPlot( 'model/chem/dend/Ca[49]', 'getConc', 'dCa49', 18 )
    addPlot( 'model/chem/spine/Ca[0]', 'getN', 'sCa0', 18 )
    addPlot( 'model/chem/spine/Ca[25]', 'getN', 'sCa25', 18 )
    addPlot( 'model/chem/spine/Ca[49]', 'getN', 'sCa49', 18 )
    addPlot( 'model/chem/psd/Ca[0]', 'getConc', 'pCa0', 18 )
    addPlot( 'model/chem/psd/Ca[25]', 'getConc', 'pCa25', 18 )
    addPlot( 'model/chem/psd/Ca[49]', 'getConc', 'pCa49', 18 )

    d = moose.vec( '/model/chem/dend/Ca' )
    s = moose.vec( '/model/chem/spine/Ca' )
    p = moose.vec( '/model/chem/psd/Ca' )
    s[5].nInit = 1000
    s[40].nInit = 5000
    moose.reinit()
    moose.start( runtime )

    fig = plt.figure( figsize = (13,10 ) )
    p1 = fig.add_subplot( 311 )
    plotVm( p1, 'dend' )
    plotVm( p1, 'head' )
    plotVm( p1, 'psd' )
    p1.legend()
    #time = numpy.arange( 0, dendVm.vector.size, 1 ) * dendVm.dt
    #p1.plot( time, dendVm.vector, label = 'dendVm' )
    p2 = fig.add_subplot( 312 )
    p2.plot( d.conc, label = 'idendCa' )
    p2.plot( s.conc, label = 'ispineCa' )
    p2.plot( p.conc, label = 'ipsdCa' )
    p2.legend()
    '''
    p2 = fig.add_subplot( 312 )
    for i in moose.wildcardFind( '/graphs/#Ca#' ):
        time = numpy.arange( 0, i.vector.size, 1 ) * i.dt
        p2.plot( time, i.vector, label = i.name )
    p2.legend()
    '''
    p3 = fig.add_subplot( 313 )
    p3.plot( getMidpts('dend'), d.conc, label = 'dendCa' )
    #p3.plot( range( 0, len( d ) ), d.conc, label = 'dendCa' )


    p3.plot( getMidpts('spine'), s.conc, label = 'spineCa' )
    p3.plot( getMidpts('psd'), p.conc, label = 'psdCa' )
    p3.legend()

    plt.show()

    '''
    '''
    app = QtGui.QApplication(sys.argv)
    #widget = mv.MoogliViewer( '/model' )
    morphology = moogli.read_morphology_from_moose( name="", path = '/model/elec' )
    widget = moogli.MorphologyViewerWidget( morphology )
    widget.show()
    return app.exec_()
    quit()

def getMidpts( compt ):
    scale = 1.0
    midpt = moose.element( '/model/chem/' + compt ).voxelMidpoint
    xpt = [i*scale for i in midpt[2*len(midpt)/3:] ]
    #for i in range( 0, len( midpt)/ 3 ):
        #xpt.append( midpt[i] * 1e6 )
    return xpt

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
