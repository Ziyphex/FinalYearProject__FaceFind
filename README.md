# Face Find Final Year Project

## Brief Introduction
This is the code base for my final year project which uses trained models to measure student engagement in web tasks. Written in Python, it works using Dlib for face detection, which functions via a Histogram of Oriented Gradients methods with a Linear Support Vector Machine.
The system then uses Dlib to perform facial landmark detection, which in turn is used to calculate the head orientation of the student.
Finally, for further insight into the students engagement level, a trained Keras model (based on the TensorFlow framework) detects the students expressions.
All data collected from this system is anonymous by default; at no point is the identity of the subject revealed, unless a requirement of the research is to do so, upon which the system will show the live feed.

## How to install
### 1.1  Install Python 3.6.1 and pip (Ubuntu)
Open a terminal and type:
```
> sudo apt-get update
> sudo apt-get install python3-pip
```
### OR
### 1.2 Install Python 3.6.1 and pip (Windows)
1. Open a browser window and navigate to the [Download](https://www.python.org/downloads/windows/) page for Windows at python.org.
2. Download Python 3.6.1 from the list according to your system requirements.
3. Run the executable after it downloads and click:
   > Install Now

### 2 Install all required libraries
Download the REQUIREMENTS.txt file and open it in an editor. Please remove the Dlib version that does not match your operating system from the list of libraries within this txt file and then run the following:
```
> pip install -r REQUIREMENTS.txt
```
It is also advised that you use a virtual environment to contain all of your python code and imports. To do this, perform the following:
```
> pip install virtualenv
> virtualenv [NAME OF ENVIRONMENT]              // fill in any name for your environment within the square brackets
```
Perform the following based on your Operating System:
```
> source [NAME OF ENVIRONMENT]/bin/activate     // used for Linux users
> .\[NAME OF ENVIRONMENT]\Scripts\activate      // used for Windows users
```


### 3 Run the program
Download the files above and unzip them to your working directory. Then type into your terminal:
```
> python face_find_app.py
```
