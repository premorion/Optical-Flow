# ${1:Optical Flow}
TODO: Write a project description
## Installation

To run the function, simply type the CW2 function with the desired inputs into the matlab command line.

You will need to put into the folder the flows.mat file in order to run the function, if not, the function will not work.

## Usage

Here is the prototype of the function:

CW2(path,prefix,first,last,digits,suffix,start);

And an example of how to use it:

CW2('E:\Onedrive\UCL\Computational Photography and Capture\Practicals\Practical6\Images','gjbLookAtTarget_',0000,0071,4,'jpg',1);

CW2.m has this inputs:

	1. path - the location or folder of the images that we want to compute, I had a number of problems with Matlab version 2014b, I had to use a absolute path instead of a relative one.
	2. prefix - the name of the images
	3. first - the number listed in the first image
	4. last - the number listed in the last image 
	5. digits - the number of digits the number has
	6. suffix - File type, in this case, the images are in jpg format
	7. start - the index of the image that we want to use as the default, in our case, the first one (1)

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D
