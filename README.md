# NJC Extract Echelle orders

By Neil Cook

Current version: 0.0.11

Date: July 11 2017


## 1. Introduction

This program is designed to take a fits image file (i.e. a CCD image) and extract out orders from an Echelle spectrograph (regardless of separation and curvature, as long as orders are distinguishable from one-another).

__IMPORTANT NOTE__

The user manual (pdf) contains images of example good and bad plots (for each of the steps in Section 3).

### 1.1 Installation and dependencies

This program was written to work with modules installed in Anaconda 2 (https://www.continuum.io/downloads) and python 2.7. It is recommended to use an installation of anaconda 2 to use this program.

Once Anaconda is installed one other module is needed (tqdm), if anaconda is installed correctly then pip can be used to install this module

```
pip install tqdm
```

#### 1. 2 Module dependencies if not using Anaconda

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

## 2. Running the Code

To run the code, change to the folder containing the ```Extract_Orders.py``` file.

This program can be configured one of four ways:
* Using default values (or modifying default values in the .py file - Not recommended)
* Modifying the values in the config file (```config.txt``` in the .py file directory)
* Entering options at run time via user interface (once python file has started)
* Entering command line options

Note that any in-putted during the running of the code will override the command line inputs which will override the config file, which will override the values in the python code

The priority is given in the following order (i.e. the lowest value overrides higher values)
*Input at run time
*Input on command line
*Input in config file
*Input in the python file

As such currently all inputs are defined in the config file and therefore changing the python file will do nothing.
To use default values just comment variables out with a ```#``` (See Section 2.3 for description of the variables) and follow instructions in the config file regarding format.

### 2.1 Using default values/Entering at run time

Type:
```commandline
python Extract_Orders.py
```

First option the program gives is:
Set up input parameters ? [ Y ] es [ N ] o :

If yes the program will display current default values, describe the values and ask if the user wishes to change the values (Values are described in Section 3.3 below.)

### 2.2 Entering command line options

This program allows values to be entered on the command line

Type:
```commandline
python Extract_Orders_py argument = value
```

where argument is selected from the list in Section 2.3 and value is the value to
be set in the program.
i.e.
```commandline
python Extract_Orders.py files="CCDimage1.fits" xblur=4 yblur=0.25
```

### 2.3 Description of variables

Variables that this program required are as follows:
1. ```filepath```  =  string, location of the folder containing fits files to extract”

2. ```plotpath```  =  string, location of place to save plots” 

3. ```savepath```  =  string, location to save extracted order fits files”

4. ```files```  =  list or string, names of image fits files to extract”

5. ```order direction```  =  string, direction of the orders.
    
    If ‘horizontal’ or ‘H’ or ‘h’ then orders are assumed to increase along the x axis
    
    If ‘vertical’ or ‘V’ or ‘v’ then orders are assumed to increase along the y axis

6. ```xblur``` = float, blur pixels in wavelength direction (to aid finding orders), if xblur and yblur are zero not Gaussian fit will be applied.

7. ```yblur``` = float, blur pixels in order direction (to aid finding orders), if xblur and yblur are zero not Gaussian fit will be applied.

8. ```pcut``` = float, the percentile to mask at, 50 is equivalent to the median (i.e. all pixels below this will not be used in making the
polynomial fit)

9. ```fitmin``` = integer, lowest order polynomial to fit

10. ```fitmax``` = integer, highest order polynomial to fit

11. ```width``` = integer, number of pixels to use for width of each order

12. ```minpixelsinorder``` = minimum number of pixels to use order (else it is discarded as noise/unusable)

## 3 The Extraction

Once the variables have been set (see Section 2) the code begins to extract the
orders.

### 3.1 Min-max clipping

The first user interface is the min-max clipping. The aim of this is to produce an
image where the orders are as clearly visible and distinguishable from the
background as possible.

The program will display the following stats in the title:

``` 
       Min max clipping
 low sigma = X high sigma = X

            Stats :
    Median is X  Std is X
  Low is at X  High is at X

```
    
and a matplotlib greyscale window of the original image. 

The low clip is by default set to the median and the high clip is set to the median plus two standard deviations (median + 2×std).

In the right hand panel two text entry boxes appear, with two buttons below "update" and "next".


Low and High are calculated using the following form:
``` 
Low = median(image) − lowsigma × std(image)

High = median(image) + highsigma × std(image)
 ``` 
the graph will refresh each time the user presses the "update" button. 
Repeat these steps until the user is satisfied with the image.

All values below ```lowsigma``` are set to 0, all values above ```highsigma``` are set to the value ‘High’.

__IMPORTANT NOTE__: 

This clipping is needed to identify the orders and
thus the user should try to produce an image where the orders are as clearly
visible and distinguishable from the background as possible (Sometimes this
requires a negative low value (indicating the user wants to use a value above
the median.


Once the user is done press "next" to go to the next step.

### 3.2 Locating Orders

The next user interface may take some time to first load (depending on the size of your CCD image and the computational power of your comptuer).

The code uses a python skimage called ```measure.label``` (see [here](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label])) algorithm to locate discrete `regions' in the modified (min-max clipping) image.

Again a UI should pop up and the title should show:

```
             Found orders
    horizontal blur = X [pixels]
    vertical blur = X [pixels]
    cut percentile = X [%]
    min pixels selected = X [pixels]
```

The blur is done in a Gaussian fashion (see [here](http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian_filter})) where:

```
sigma = [horizontal blur, vertical blur]
```

The cut is then applied using the `percentile' (pcut) variable. This creates a mask:

```
Mask = 1  if pixel values > percentile
Mask = 0  otherwise
```

and additional requirement is that each "blob" as "min pixels selected" number of pixels in its region. If it has less it will not be identified as a useful "blob".
 
This mask is used to define unique regions (where touching pixels values = 1). Thus one should choose values of horizontal blur, vertical blur  and percentile such that the orders are separated (the graph displayed will show the regions found).

To update the values use the "update" button (again this may take some time to process). To accept changes click the "next" button.

### 3.3 Removing unwanted orders

This loads up another user interface with the fitted orders shown in red. 

Orders are calculated by fitting a polynomial fit (```numpy.polyfit```) to each order (for powers between "fitmin" and "fitmax", see Section 2.3) the chosen polynomial fit is the polynomial with the lowest chi squared value.

The user interface will have one text entry box and three buttons. To remove orders list them in the text entry box (separated by a white space or a comma). All orders that the user wishes to remove must be entered, deleting an order from this text entry box will re-add it to the kept orders.

Pressing the "update" button will update the figure with the remaining orders. "Start fitting process again" will restart all steps (from Section 3.1 onwards). Pressing the "Accept Orders" button will proceed to running the extraction with the remaining orders (currently shown in the figure). You __must__ click "update" to remove orders __before__ clicking "Accept Orders".

__IMPORTANT__: 

Look out for stray orders in the middle or at the edges of the graph) one can simply start the extraction again or remove some badly fitted or incorrect orders.

### 3.4 The outputted, extracted spectra

The program will produce two types of output. 

1) It will produce "pixel number" vs "counts" graphs and fits files for each extracted order (in the location defined in "plotpath" and "savepath" respectively, see Section 2.3).
2) It will produce fits files which have columns: "pixel number" and "counts".

## 4 Extracting from previously fitted image

Once you have run the code once you may choose to use the polynomial fits from a previous image (henceforth the old image) to fit a new image. This can be done by running the program ```Extract_Orders_from_polyfits.py```.

This code essentially accesses parts of the ```Extract_Order.py``` code and just runs the last steps wihtout the user interface (see Section 3).

One needs to set the parameters for "polypath" such that it links to the polyfit file created in the previous run (i.e. in ```./Savedfits/All_saved_polynomials.fits```) with the old image and set the "width" and "files" variables as in Section 2. 

This can be done by editing the program or by modifying them on run time (again in Section 2).

The outputs will save in a new folder (defined by the name of the file) and no polyfits file will be saved.