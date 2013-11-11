roi_browser
===========

Python browser to quickly flip through directory of interferograms

You need a functional python installation to run the browser. I use [Enthought's Canopy Software](https://www.enthought.com/products/canopy/), but as long as you have the following, it should work:

* python 2.7.3
* matplotlib 1.3.0
* numpy 1.7.1

Usage
-----
To run the browser on your machine do the following:

	cd install-here
	git clone https://github.com/scottyhq/roi_browser
	echo PATH=$PATH:install-here/roi_browser
	roi_browser.py 'int*/filt*unw'
