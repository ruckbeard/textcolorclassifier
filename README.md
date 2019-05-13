# textcolorclassifier
A machine learning model to output whether text should be white or black based on an algorithm. Model was trained on data generated by the algorithm and then loaded into a Flask application to show output in a user friendly interface.

#Training
The script model.py will train the model based on the CSV data in data.txt which contains a list of RGB values and text color value 1 for white and 0 for black.

Running the script will create an output file that will be loaded by predict.py which loads the model and takes an RGB value and outputs either 1 or 0 for white or black.

#Demo
You can see it working live here:
http://ec2-54-235-14-179.compute-1.amazonaws.com/predict

You can also input your own values by entering them through the url
http://ec2-54-235-14-179.compute-1.amazonaws.com/predict/{red}/{green}/{blue}

for example:
http://ec2-54-235-14-179.compute-1.amazonaws.com/predict/240/140/100
