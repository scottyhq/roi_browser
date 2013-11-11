#!/usr/bin/env python
"""
Simple python viewer to browse phase data

Usage: browse_interferogram.py "./int*/filt*unw"
NOTE: name argument must be in quotes
'n' for next interferogram & 'p' for previous or drag mouse along slider

Author: Scott Henderson
Requires:
    python 2.7
    matplotlib 1.3
    
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import os
import re
import glob
#from matplotlib.colors import LogNorm # for radar amplitude data

__version__ = '0.2'


def format_coord(x, y):
    """ print image value in additionto x,y coordinate"""
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, phs=%1.4f'%(x, y, z)
    #elif col>=0 and col<numcols and row>=0 and row<numrows:
    #    z = X[row,col]
    #    return 'x=%1.4f, y=%1.4f, amp=%1.4f, phs=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)


def get_files(filetype):
    """Make a dictionary of dates:filenames"""
    igramsDict = {}
    path = os.path.expanduser(filetype) # make sure ~ in path is expanded
    paths = glob.glob(path)
    for path in paths:
		#print path
		filename = os.path.basename(path)
		datePair = re.search("\d{6}-\d{6}",filename).group()
		igramsDict[datePair] = path
    return igramsDict


def load_rsc(path):
    """Read metadata from .rsc file into python dictionary"""
    #print path
    metadata = {}
    rsc = open(path + '.rsc', 'r')
    # Remove blank lines, retaining only lines with key, value pairs
    allLines = [line for line in rsc.readlines() if line.strip()]
    for line in allLines:
        var, value = line.split()
        metadata[var] = value
    rsc.close()
    #return (int(metadata['FILE_LENGTH']), int(metadata['WIDTH']))
    return metadata


def load_data(igramPath, args):
    ''' call approprate function to load ROI_PAC image data'''
    metadata = load_rsc(igramPath)
    
    #NOTE: would be good if 'baseline' were included
    if args.verbose:
        print 'Path: {PATH}\nTimespan: {TIME_SPAN_YEAR}\nLength: {FILE_LENGTH}\nWidth: {WIDTH}'.format(*metadata)
    
    dims = (int(metadata['FILE_LENGTH']), int(metadata['WIDTH']))
    if igramPath.endswith('int'):
	amp,phs = load_cpx(igramPath,dims)
    else:
        amp,phs = load_phase(igramPath, dims)

    if args.displacement and igramPath.endswith('unw'):
        phs = phs * metadata['WAVELENGTH'] / (4*np.pi)
    else:
        print 'can only convert unwrapped files to displacement [m]'
        

    return amp,phs


def load_phase(igramPath, dims):
	"""Load phs array from .unw file"""	
	#NOTE: could print this out if 'verbose' is selected
	#print igramPath, dims

	datatype = np.dtype('<f4')
		
	phs = np.zeros(dims, dtype=datatype)
	amp = phs.copy()

	with open(igramPath,'rb') as f:
		rows = range(dims[0]-1)
		# Set offset to read phase array
		junk = np.fromfile(f, dtype=datatype, count=dims[1]) 
		for i in rows:
			phs[i,:] = np.fromfile(f, dtype=datatype, count=dims[1])
			amp[i,:] = np.fromfile(f, dtype=datatype, count=dims[1])
		
	phs[phs==0] = np.NaN
	amp[amp==0] = np.nan
	# Reorient arrays to North up, West left
	phs = np.rot90(phs,2)
	amp = np.rot90(amp,2)
	
	return amp,phs


def load_cpx(igramPath, dims):
	""" for loading int or slc files """
	data = np.fromfile(igramPath,dtype='complex64')
	data = np.reshape(data, dims) #length,width
	
	amp = np.abs(data)
	phs = np.angle(data)

	#print amp
	phs[phs==0]= np.nan
	amp[amp==0]= np.nan
	
	# Reorient arrays to North up, West left
	phs = np.rot90(phs,2)
	amp = np.rot90(amp,2)

	return amp,phs


def create_single_browser(igramsDict, args):
	""" Just show phase """
	order = igramsDict.keys()
	order.sort()
	dates = order[0]
	igPath = igramsDict[dates]
    
        junk,data = load_data(igPath, args)
    
        # Platform-independent Slider to go through interferograms
	fig = plt.figure(figsize=(8.5,11))
	title = fig.suptitle(igPath, fontsize=14, fontweight='bold')
        #fig.suptitle(os.path.dirname(igPath), fontsize=14, fontweight='bold')
	#fig.suptitle(os.path.basename(os.getcwd()), fontsize=14, fontweight='bold')
	ax = plt.subplot(111)
	#plt.subplots_adjust(left=0.25, bottom=0.25)
	#ac.title(os.path.basename(igPath))
	im = plt.imshow(data, origin='lower',cmap=plt.cm.jet)
	im.set_extent([-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5])
	cb = plt.colorbar()    
	
	# Print array value in addition to cursor location
	ax.format_coord = format_coord 
	
	#slider set-up
	axcolor = 'lightgoldenrodyellow'
	axIG = plt.axes([0.25, 0.02, 0.65, 0.03], axisbg=axcolor) #left,bottom, width, height
	sIG = Slider(axIG, 'IG #/{0}'.format(len(order)), 1, len(igramsDict),
	               valinit=1,
	               valfmt='%i',
	               closedmin=True,
	               closedmax=True,
	               dragging=True)
	
	def update(val):
		dates = order[int(sIG.val)-1]
		igPath = igramsDict[dates]
		amp,data = load_data(igPath, args)
		im.set_data(data)
		im.autoscale() # autoscale colorbar to new data
		ax.relim()
		ax.autoscale_view(tight=True)
		#ax.autoscale_view(True,True,True)
		im.set_extent([-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5])
		title.set_text(igPath)
		#ax.set_title(os.path.basename(igPath))
		plt.draw()
        
	def onpress(event):
		if event.key not in ('n', 'p', 'left', 'right'): return
		if event.key=='n' or event.key=='right': 
		    newval = sIG.val + 1
		elif event.key=='p' or event.key=='left':
		    newval = sIG.val - 1
		if newval < sIG.valmin: 
		    newval = sIG.valmax
		if newval > sIG.valmax: 
		    newval = sIG.valmin
		sIG.set_val(newval) # update() automatically called
	
	fig.canvas.mpl_connect('key_press_event', onpress)
	sIG.on_changed(update)
	plt.show()


def create_double_browser(igramsDict, args):
	""" show amplitude and phase side-by side """
	print igramsDict
	order = igramsDict.keys()
	order.sort()
	dates = order[0]
	igPath = igramsDict[dates]
	print igPath
	
	fig = plt.figure(figsize=(11,8.5)) #landscape
	title = fig.suptitle(igPath, fontsize=14, fontweight='bold') 
	#plt.title(os.path.basename(igPath[0]))
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (1, 2),
                direction="row",
                axes_pad = 1.0,
                add_all=True,
                label_mode = 'all', #'all', 'L', '1'
                share_all = True,
                cbar_location='right', #top,right
                cbar_mode='each', #each,single,None
                cbar_size=0.1,#"7%",
                cbar_pad=0.0#,"1%",
                )

        ax1 = grid[0]
        ax2 = grid[1]
        amp,phs = load_data(igPath, args)
        im1 = ax1.imshow(amp, origin='lower',
                            cmap=plt.cm.gray,
                            extent=[-0.5,amp.shape[1]-0.5,amp.shape[0]-0.5,-0.5],
                            #norm=LogNorm(vmin=np.nanmin(amp),vmax=np.nanmax(amp)) #logrithmic scale
                            )
        ax1.cax.colorbar(im1, format='%0.0e') #NOTE: can pass format=ScalarFormatter here.
            
        im2 = ax2.imshow(phs, origin='lower',cmap=plt.cm.jet)        
        im2.set_extent([-0.5, phs.shape[1]-0.5, phs.shape[0]-0.5, -0.5])
        cb2 = ax2.cax.colorbar(im2)
        #cb2.set_label('phase') 

	#slider set-up
	axcolor = 'lightgoldenrodyellow'
	axIG = plt.axes([0.20, 0.05, 0.60, 0.02], axisbg=axcolor) #left, bottom, width, height
	sIG = Slider(axIG, 'IG #/{0}'.format(len(order)), 1, len(igramsDict),
	               valinit=1,
	               valfmt='%i',
	               closedmin=True,
	               closedmax=True,
	               dragging=True)
	
	def update(val):
		dates = order[int(sIG.val)-1]
		igPath = igramsDict[dates]
		amp,phs = load_data(igPath, args)
		
		im1.set_data(amp)
		
		im1.autoscale() # autoscale colorbar to new data
		ax1.relim()
		ax1.autoscale_view(tight=True)
		#ax.autoscale_view(True,True,True)
		im1.set_extent([-0.5, amp.shape[1]-0.5, amp.shape[0]-0.5, -0.5]) #may need to move
		
		im2.set_data(phs)
		
		im2.autoscale() # autoscale colorbar to new data
		ax2.relim()
		ax2.autoscale_view(tight=True)
		im2.set_extent([-0.5, phs.shape[1]-0.5, phs.shape[0]-0.5, -0.5])
		
		title.set_text(igPath)
		plt.draw()
        
	def onpress(event):
		#print event.key
		if event.key not in ('n', 'p', 'right', 'left'): return
		if event.key=='n' or event.key=='right': 
		    newval = sIG.val + 1
		elif event.key=='p' or event.key=='left':  
		    newval = sIG.val - 1
		
		if newval < sIG.valmin: 
		    newval = sIG.valmax
		if newval > sIG.valmax: 
		    newval = sIG.valmin
		
		sIG.set_val(newval) # update() automatically called
	
	fig.canvas.mpl_connect('key_press_event', onpress)
	sIG.on_changed(update)
	plt.show()


def main():
    parser = argparse.ArgumentParser(description='browse ROI_PAC images in a directory')
    # Positional arguments
    parser.add_argument('files', help='file string with wildcards (e.g. "int*/filt*32*int")')
    # Optional arguments
    #parser.add_argument('-m','--cmap', default='jet', help='matplotlib or basemap colormap string')
    #parser.add_argument('-c','--clim', type=float, default=(None,None), nargs=2, metavar=('cmin', 'cmax'), help='manual limits for colorbar') #default is none if not
    parser.add_argument('-a','--amp', action='store_true', default=False, help='show radar amplitude alongside phase')
    parser.add_argument('-d','--displacement', action='store_true', default=False, help='convert unwrapped phase to displacement [m]')
    #parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("-v", "--verbose", default=False, help="increase output verbosity", action="store_true")
    parser.add_argument('--version', action='version', version='{0}'.format(__version__))
    
    
    args = parser.parse_args()
             
    igrams = get_files(args.files)
    #print igrams
    
    print '\n Use right and left arrow keys to change cycle images. Or click on the slider.\n '
    if args.amp:
        create_double_browser(igrams, args)
    else:
        create_single_browser(igrams, args)




if __name__ == '__main__':
    main()
