roi_browser
===========

Python browser to flip through directory of interferograms (.int or .unw)

*Be warned*, it can be a bit slow for full resolution images, but it's super useful! 

You need a functional Python installation to run the browser. I recommend using the free [Continuum Anaconda Distribution](https://store.continuum.io/cshop/anaconda/). Using the *conda* command line tool, you can easily ensure the proper dependencies are installed (see instructions below).

Installation
------------
To run the browser on your machine do the following:

	git clone https://github.com/scottyhq/roi_browser
	cd roi_browser
	conda env create
	
To link an executable run the following (adjusting directories as needed):
	
	ln -s /path-to-roi_browser/roi_browser.py /user/local/bin/roi_browser 
	

Usage
_----
	
	source activate roi_browser
	roi_browse --help
	roi_browser.py 'roi_browser_pathexample/int*/filt*unw'

You should then see a window like this:
![Screenshot](/screen_shot.png)


Troubleshooting
---------------
See [this useful introduction and guide](http://continuum.io/blog/conda-data-science) to conda Python environments. To return to your default environment run:
	
	source deactivate

By default the browser uses the 'TkAgg' matplotlib backend since it is the default. You can change to another by modifying line #10 in the roi_browser.py source code (e.g. matplotlib.use('Qt'))
