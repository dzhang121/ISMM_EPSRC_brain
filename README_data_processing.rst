Quantitative monitoring and modelling of retrodialysis drug delivery in a brain phantom
=======================================================================================

This is the dataset and software supporting the study *Quantitative 
monitoring and modelling of retrodialysis drug delivery in a brain phantom*
by Etienne Rognin, Niamh Willis-Fox, Ronan Daly, Institute for Manufacturing, 
Department of Engineering, University of Cambridge, 17 Charles Babbage Road, 
Cambridge CB3 0FS, United Kingdom.


License
-------
Copyright (C) 2022 by Etienne Rognin <ecr43 at cam.ac.uk>, Niamh Willis-Fox, 
Ronan Daly <rd439 at cam.ac.uk>.

``Tiff`` images fall under the Creative Commons Attribution 4.0 license. 
You must give appropriate credit, provide a link to the license, and 
indicate if changes were made. You may do so in any reasonable manner, but 
not in any way that suggests the licensor endorses you or your use. 

Software (for example ``.ipynb`` IPython Notebook files) fall under BSD 
Zero clause license below:

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.



Contents
--------

There are two main folders:

 1. ``example`` which contains all the relevant scripts and a small dataset
    of raw images. Use this for a minimal example.
   
 2. ``supporting data`` contains the data and software used to support the 
    present study. There is one folder per experiment. The naming convention 
    is ``date flow-rate``. Usage of each case folder is the same as the main 
    ``example`` folder. There is also a ``model`` folder containing scripts
    to analyse the whole dataset and for special cases described in the 
    paper.


Usage
-----

Raw images are stored in zip files which must be unzipped first before 
running data analysis scripts.

    
The software is a set of self-documented IPython Notebook files. The use of
these scripts requires the following to be installed:
  - Python 3.7+
  - Jupyter Notebook
  - Python modules: numpy, skimage, matplotlib, ipywidgets, tqdm, pywt, 
    natsort, abel
    
For each experimental folder, three scripts must be run sequentially:

 1. ``1 Preprocessing.ipynb`` loads raw images and applies filtering. The 
    output is absorbance fields (``A.npy`` Python Numpy file) and probe
    mask (``mask.npy``). Coordinates of the area of interest are stored in
    Python file ``cuvette.pickle``.
 
 2. ``2 Integrate.ipynb`` measures the mass seen in the image stack 
    discarding any probe shadow. The output is a text file ``mass.txt``.
    
 3. ``3 Reconstruct.ipynb`` computes concentration fields using the inverse
    Abel transform and applies a correction to the total mass measured. The
    output is a text file ``mass_Abel.txt``, and concentration fields 
    (optional).
    
Additionally, one can generate videos from images by running the script
named ``tiff2mp4.sh``, which requires ffmpeg to be installed. 

