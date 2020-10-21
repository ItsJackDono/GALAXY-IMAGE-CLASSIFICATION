# GALAXY-IMAGE-CLASSIFICATION-USING-MACHINE-LEARNING-IN-PYTHON

The project developed was for my university disseration with the focus of using a Deep Convolution Neural Network on a set of galaxy images. The galaxy images are classified using a weighted probability distribution of 37 decisions which ditctate whether an image is of a type of class (Spiral, Elipitcal, and Irregular). The dataset trained used the galaxy zoom challenge hosted on kaggle (https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). The CNN architecture design was developed with hardware constraints in mind with two prototype algorithms developed with a final testing algoirthm which allowed for 60%+ accruacy depending on the amount of Epoch iterations. Images and documentation can be found within the repository with results of testing and design methods chosen.

For running the project please download the files and follow the steps below for an indepth description of how to set up and execute  the project.

#Setting up the project for execution

Setting up a model requires a few steps to first run the python scripts. First the datasets used where downloaded from
 https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data 
Required datafiles: 
•	Images_training
•	Solutions_training
•	Images_test
•	All_zeros_benchmark 

After downloading the 4 datafiles simply place in a directory that can be accessed and retrieved from inside the python scripts.
After Downloading the ZIP file containing the artefact 3 files should be present,
•	GClassPrototype1.py
•	GClassPrototype2.py
•	testingPrototype2.py
In the Same folder space create an empty models.h5 file to store the data after running “testingPrototype2.py”.

This is run on python 3.7 and windows 10 operating system. Other python versions may contain errors. At the top of each file the necessary library’s need to be imported if not already done so. The three python scripts operate as follows:
•	GClassPrototype.py simply runs the model by training the Images_training and Solutions_training. Replace the directory of the downloaded data files into the same directory’s present in the python script.
•	GClassPrototype2.py must be run to train the models training and validation weights the H5PY file.
•	After these have been trained the testingPrototype2.py script can be run on testing data. Make sure to change the test directory to Images_test directory.
•	Make sure  to All_zeros_benchmark directory is correct. 
If any further testing is needed for the testingPrototype.py new empty H5PY files are needed.
