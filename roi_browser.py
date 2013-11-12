#!/usr/bin/env python
"""
Browse though interferograms in a directory

Usage: roi_browser.py "./int*/filt*unw"
Note: designed for 'int' and 'unw' files... but wouldn't take much effort to handle others...
Author: Scott Henderson (st54@cornell.edu)
Requires:
    * python 2.7
    * matplotlib 1.3
    * numpy 1.7
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, MultiCursor
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import argparse
import os
import re
import glob
from matplotlib.colors import LogNorm # for radar amplitude data

__version__ = '0.2'
#print __version__

def format_coord(x, y):
	"""Prints pixel value in additionto x,y coordinate
	"""
	#X = plt.gca().get_images()[0].get_array() #doesn't work b/c image & slider also axes
	#X = plt.gcf().get_axes()[0].get_images()[0].get_array()
	#X = self.get_images()[0].get_array()
	X = np.flipud(plt.gcf().get_axes()[0].get_images()[0].get_array()) #messy but works!
	numrows,numcols = X.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>=0 and col<numcols and row>=0 and row<numrows:
		z = X[row,col] #must pass X
		return 'x=%1.1f, y=%1.1f, phs=%1.3f'%(x, y, z)
	#else:
	#	return 'x=%1.3f, y=%1.3f'%(x, y)

def format_coord2(x,y):
	Amp = np.flipud(plt.gcf().get_axes()[0].get_images()[0].get_array()) #messy but works!
	Phs = np.flipud(plt.gcf().get_axes()[1].get_images()[0].get_array())
	numrows,numcols = Amp.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>=0 and col<numcols and row>=0 and row<numrows:
		amp = Amp[row,col]
		phs = Phs[row,col]
		return 'x=%1.1f, y=%1.1f, amp=%1.3f, phs=%1.3f' % (x, y, amp, phs) 


def get_files(filetype):
    """Make a dictionary of dates:filenames
	"""
    igramsDict = {}
    path = os.path.expanduser(filetype) # make sure ~ in path is expanded
    paths = glob.glob(path)
    for path in paths:
		filename = os.path.basename(path)
		datePair = re.search("\d{6}-\d{6}",filename).group()
		igramsDict[datePair] = path
    return igramsDict


def load_rsc(path):
    """Read metadata from .rsc file into python dictionary
	"""
    metadata = {}
    rsc = open(path + '.rsc', 'r')
    # Remove blank lines, retaining only lines with key, value pairs
    allLines = [line for line in rsc.readlines() if line.strip()]
    for line in allLines:
        var, value = line.split()
        metadata[var] = value
    rsc.close()
    return metadata


def load_data(igramPath, args):
    """calls approprate function to load ROI_PAC image data"""
    metadata = load_rsc(igramPath)
    
    if args.verbose:
			print 'Path: {0}\nTimespan: {TIME_SPAN_YEAR}\nLength: {FILE_LENGTH}\nWidth: {WIDTH}\n'.format(igramPath,**metadata)
    
    dims = (int(metadata['FILE_LENGTH']), int(metadata['WIDTH']))
    if igramPath.endswith('int'):
	amp,phs = load_cpx(igramPath,dims)
    else:
        amp,phs = load_phase(igramPath, dims)
	
	if args.displacement:
		phs = phs * float(metadata['WAVELENGTH']) / (4*np.pi)
        

    return amp,phs


def load_phase(igramPath, dims):
	"""Load phase array from .unw file
	"""	
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
	"""Loads complex number arrays (int or slc files)
	"""
	data = np.fromfile(igramPath,dtype='complex64')
	data = np.reshape(data, dims) 
	amp = np.abs(data)
	phs = np.angle(data)
	phs[phs==0]= np.nan
	amp[amp==0]= np.nan
	
	# Reorient arrays to North up, West left
	phs = np.rot90(phs,2)
	amp = np.rot90(amp,2)

	return amp,phs


def create_single_browser(igramsDict, args):
	"""Just show phase data
	"""
	order = igramsDict.keys()
	order.sort()
	dates = order[0]
	igPath = igramsDict[dates]

	junk,data = load_data(igPath, args)
    
	fig = plt.figure(figsize=(8.5,11))
	titlestr = os.path.basename(igPath)
	title = fig.suptitle(titlestr, fontsize=14, fontweight='bold')
	ax = plt.subplot(111)
	im = plt.imshow(data, origin='lower',cmap=plt.cm.jet)
	im.set_extent([-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right',size='5%',pad=0.05)
	cb = plt.colorbar(im, cax=cax)    
	if args.displacement:
		cb.set_label('dlos [m]')
	else:
		cb.set_label('dlos [rad]') 


	# Print array value in addition to cursor location
	ax.format_coord = format_coord 
	
	#Set up matplotlib Slider widget
	axcolor = 'lightgoldenrodyellow'
	axIG = plt.axes([0.25, 0.02, 0.65, 0.03], axisbg=axcolor) 
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
		im.autoscale()
		ax.relim()
		ax.autoscale_view(tight=True)
		im.set_extent([-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5])
		title.set_text(os.path.basename(igPath))
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
		sIG.set_val(newval) 
	
	fig.canvas.mpl_connect('key_press_event', onpress)
	sIG.on_changed(update)
	plt.show()


def create_double_browser(igramsDict, args):
	"""Show amplitude and phase side-by side 
	"""
	#print igramsDict
	order = igramsDict.keys()
	order.sort()
	dates = order[0]
	igPath = igramsDict[dates]
	#print igPath
	
	fig = plt.figure(figsize=(11,8.5)) 
	titlestr = os.path.basename(igPath)
	title = fig.suptitle(titlestr, fontsize=14, fontweight='bold') 
	
	grid = ImageGrid(fig, 111, 
                nrows_ncols = (1, 2),
                direction="row",
                axes_pad = 1.0,
                add_all=True,
                label_mode = 'all', 
                share_all = True,
                cbar_location='right', 
                cbar_mode='each', 
                cbar_size=0.1,
                cbar_pad=0.0
                )
	ax1 = grid[0]
	ax2 = grid[1]
	amp,phs = load_data(igPath, args)
	
	# If amplitude spans orders of magnitude, convert to log scale...
	if np.abs(np.nanmax(amp)/np.nanmin(amp)) > 1000.0:
		norm = LogNorm(vmin=np.nanmin(amp),vmax=np.nanmax(amp))
		fmt = '%0.0e'
	else:
		norm = None
		fmt = '%0.2f'

	im1 = ax1.imshow(amp, origin='lower',
                            cmap=plt.cm.gray,
                            extent=[-0.5,amp.shape[1]-0.5,amp.shape[0]-0.5,-0.5],
                            norm=norm)                         
	ax1.cax.colorbar(im1, format=fmt)
	
	im2 = ax2.imshow(phs, origin='lower',cmap=plt.cm.jet)
	im2.set_extent([-0.5, phs.shape[1]-0.5, phs.shape[0]-0.5, -0.5])
	cb2 = ax2.cax.colorbar(im2)
	
	if args.displacement:
		ax2.cax.set_label('dlos [m]')
	else:
		ax2.cax.set_label('dlos [rad]') 
	
	# Add cursor
	if args.cursor:
		multi = MultiCursor(fig.canvas, (ax1,ax2), color='k', lw=1, horizOn=True, vertOn=True)

	# Print array values on screen
	ax1.format_coord = format_coord2
	ax2.format_coord = format_coord2

	#Set up matplotlib Slider widget
	axcolor = 'lightgoldenrodyellow'
	axIG = plt.axes([0.20, 0.05, 0.60, 0.02], axisbg=axcolor) 
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
		
		im1.autoscale() 
		ax1.relim()
		ax1.autoscale_view(tight=True)
		im1.set_extent([-0.5, amp.shape[1]-0.5, amp.shape[0]-0.5, -0.5]) 
		
		im2.set_data(phs)
		
		im2.autoscale() 
		ax2.relim()
		ax2.autoscale_view(tight=True)
		im2.set_extent([-0.5, phs.shape[1]-0.5, phs.shape[0]-0.5, -0.5])
		
		title.set_text( os.path.basename(igPath) )
		plt.draw()
        
	def onpress(event):
		if event.key not in ('n', 'p', 'right', 'left'): return
		if event.key=='n' or event.key=='right': 
		    newval = sIG.val + 1
		elif event.key=='p' or event.key=='left':  
		    newval = sIG.val - 1
		
		if newval < sIG.valmin: 
		    newval = sIG.valmax
		if newval > sIG.valmax: 
		    newval = sIG.valmin
		
		sIG.set_val(newval) #update() automatically called
	
	fig.canvas.mpl_connect('key_press_event', onpress)
	sIG.on_changed(update)
	plt.show()


def main():
	"""Parse command line arguments
	"""
	parser = argparse.ArgumentParser(description='browse ROI_PAC images in a directory')
	# Positional arguments
	parser.add_argument('files', help='file string with wildcards (e.g. "int*/filt*32*int")')
	
	# Optional arguments
	#parser.add_argument('-m','--cmap', default='jet', help='matplotlib or basemap colormap string')
	#parser.add_argument('-c','--clim', type=float, default=(None,None), nargs=2, metavar=('cmin', 'cmax'), help='manual limits for colorbar') #default is none if not
	parser.add_argument('-c','--cursor',action='store_true',default=False,help='show cursorin both amp and phs windows (need to also use -a)')
	parser.add_argument('-a','--amp', action='store_true', default=False, help='show radar amplitude alongside phase')
	parser.add_argument('-d','--displacement', action='store_true', default=False, help='convert unwrapped phase to displacement [m]')
	#parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument("-v", "--verbose", default=False, help="increase output verbosity", action="store_true")
	parser.add_argument('--version', action='version', version='{0}'.format(__version__))
	
	args = parser.parse_args()
	
	if args.displacement==True and not args.files.endswith('unw'):
		print 'Warning: can only convert unwrapped files to displacement... [m]'
		args.displacement=False
	
	igrams = get_files(args.files)
	#print igrams
	print '\n=================== roi_browser ===========================\n
	print '\n Use right and left arrow keys to change cycle images. Or click on the slider.\n '
	if args.amp:
		create_double_browser(igrams, args)
	else:
		create_single_browser(igrams, args)

if __name__ == '__main__':
    #print 'new version'
	main()
