# NJC Extract Echelle orders

By Neil Cook

Current version: 0.0.11

Date: July 11 2017


## 1. Introduction

This program is designed to take a fits image file (i.e. a CCD image) and extract out orders from an Echelle spectrograph (regardless of separation and curvature, as long as orders are distinguishable from one-another).

### 1.1 Installation and dependencies

This program was written to work with modules installed in Anaconda 2 (https://www.continuum.io/downloads) and python 2.7. It is recommended to use an installation of anaconda 2 to use this program.

Once Anaconda is installed one other module is needed (tqdm), if anaconda is installed correctly then pip can be used to install this module

```
pip install tqdm
```

#### Module dependencies if not using Anaconda

This program relies on the following modules in python 2.7 (with earliest tested version listed)


* numpy (1.11.2)
* astropy (1.1.2)
* matplotlib (1.5.1)
* tqdm (4.4.0)
* skimage (0.12.3)
* Tkinter

Or in python 3.4

* numpy (1.12.1)
* astropy (1.3.2)
* matplotlib (2.0.2)
* tqdm (4.11.2)
* tkinter
* skimage (0.13.0)

Other versions may work but these are the ones tested.

## Running the Code

Insert text here