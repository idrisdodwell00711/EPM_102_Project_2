Project 2, EPM102. Object detector.

Git clone or download the code into your own repo.

Run - pip install open-cv or pip3 install opencv-python on the terminal. For trouble shooting see - https://stackoverflow.com/questions/51853018/how-do-i-install-opencv-using-pip
Run - pip install numpy

Run - python index.py

Make sure you are using Python 3.9.
To set up a virtual conda enviroment on VS code, control P, then select python 3.9 and conda env.

The code will display all the objects, their matches, and a bounding box with the object's name. The method used returns the X and Y coordinates of the bounding box.
Although this is not used, it is printed off. To access the coordinates please save the object's method call (e.g. processor.process_query_images()) to a variable 
(i.e. x_y_BB = processor.process_query_images(query_images, train_image, train_bboxes, object_names, query_objects)).

The project started off by importing the open cv in python and the loading the training images which has the object of interest in it. 
Then used image matching algorithm to extract the features by keypoints detection and descriptors computation. 
A bounding box was applied to these area of interest so to make the algorithm focus only on those set of features and not the other features in the background.
After this a mask was created for the training image based on the bounding box co-ordinates, 
this masked training image then helps the algorithm to focus on detecting features using FLANN, 
the goal here was to find correspondences between the keypoints in the training image and the query image, 
and hence successfully detecting the right object in the image.
