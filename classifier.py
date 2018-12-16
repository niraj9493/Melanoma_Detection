'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from http://mlr.cs.umass.edu/ml/datasets/Spambase
It contains pre-processed metrics, such as the frequency of certain
words and letters, from a collection of emails. A classification for
each one indicating 'spam' or 'not spam' is in the final column.
See the linked page for full details of the data set.

This script uses three classifiers to predict the class of an email
based on the metrics. These are not representative of modern spam
detection systems.
'''

# Remember to update the script for the new data when you change this URL
URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/spambase/spambase.data"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os 
import sys
from skimage.feature import hog

# =====================================================================
def preprocess(image):
    clahe = cv.createCLAHE(3.0, (8,8))
    color_channels = [];
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    color_channels = cv.split(image)
    color_channels[0] =  clahe.apply(color_channels[0])
    image = cv.merge(color_channels)
    return cv.cvtColor(image, cv.COLOR_YCrCb2BGR)

def Segment(image):
    ch = [ i for i in cv.split(image)]

    for i in range(len(ch)):
        ret, ch[i] = cv.threshold(ch[i],0,255,cv.THRESH_BINARY_INV +cv.THRESH_OTSU)

    mask = ch[0].astype(np.uint32) + ch[1].astype(np.uint32) + ch[2].astype(np.uint32)
    mask = mask/3
    mask = mask.astype(np.uint8)
    _,mask = cv.threshold(mask,85,255,cv.THRESH_BINARY)
    return mask

def feature_extract(img, mask):
    hog = cv.HOGDescriptor((640,480), (16,16), (8,8), (8,8), 9, 4, 0.2, 1, 64)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel=cv.getStructuringElement(cv.MORPH_RECT,(4,4)))
    mask = cv.medianBlur(mask,7)
    mask = cv.medianBlur(mask,5)
    mask = cv.medianBlur(mask,3)
    mask,conts,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    mask = img * mask[:,:,None].astype(img.dtype)
    mask = cv.drawContours(mask, conts, -1, (0,255,0), 3)
    #fd, hog_image = hog(mask, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    return hog.compute(mask)


def get_features_and_labels():
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Replace missing values with 0.0, or we can use
    # scikit-learn to calculate missing values (below)
    #frame[frame.isnull()] = 0.0

    # Convert values to floats
    #arr = np.array(frame, dtype=np.float)

    X = []
    y = []
    p_size = 0
    n_size = 0
    dir_p = r"G:/Melanoma Detection/data/data_positive/"
    dir_n = r"G:/Melanoma Detection/data/data_negative/"
    for f in os.listdir(dir_p):
       if f.endswith('.jpg'):
           #files.append(dir_p+f)
           img = cv.imread(dir_p+f)
           img = cv.resize(img,(640,480))
           if img is None:
                raise Exception("No Image Loaded !")
           img = preprocess(img)
           X.append(feature_extract(img,Segment(img)))
           #y.append(np.array(1))
           y.append(1)
           #TrainingSamples.append(feature_extract(img,Segment(img)))
           #label.append(1)

    for f in os.listdir(dir_n):
       if f.endswith('.jpg'):
           #files.append(dir_n+f)
           img = cv.imread(dir_n+f)
           img = cv.resize(img,(640,480))
           if img is None:
                raise Exception("No Image Loaded !")
           img = preprocess(img)
           X.append(feature_extract(img,Segment(img)))
           y.append(-1)
           #TrainingSamples.append(feature_extract(img,Segment(img)))
           #label.append(-1)
    # Use the last column as the target value
    #X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
    
    # Use 80% of the data for training; test against the rest
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn.pipeline.make_pipeline could also be used to chain 
    # processing and classification into a black box, but here we do
    # them separately.
    
    # If values are missing we could impute them from the training data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    from joblib import dump
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    X_train = np.array(X_train)
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))

    X_test = np.array(X_test)
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    
    scaler.fit(X_train)
    dump(scaler, 'Scaler.xml')
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.fit_transform(y_test)
    
    # Return the training and test sets
    return X_train, X_test, y_train, y_test


# =====================================================================

def classify(image, svm):
    image = cv.resize(image,(640,480))
    image = preprocess(image)
    hog = feature_extract(image,Segment(image))

    hog = np.array(hog)
    nx, ny = hog.shape
    hog = hog.reshape(nx*ny)
    from sklearn.preprocessing import StandardScaler
    from joblib import load
    scaler = load('Scaler.xml')
    hog = scaler.transform([hog])
    return svm.predict(hog)

def evaluate_classifier(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different classifiers to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, precision, recall)
    for each learner.
    '''

    # Import some classifiers to test
    from sklearn.svm import LinearSVC, NuSVC
    from sklearn.ensemble import AdaBoostClassifier

    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import precision_recall_curve, f1_score
    from joblib import dump
    # Here we create classifiers with default parameters. These need
    # to be adjusted to obtain optimal performance on your data set.
    
    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    
    # Save the classifier
    dump(classifier, 'LinearSVC.xml')
    print("Linear Testing Predictions")
    print(classifier.predict(X_test))
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)
    # Fit the classifier
    classifier.fit(X_train, y_train)

    # Save the classifier
    dump(classifier, 'NuSVC.xml')
    print("Nu Testing Predictions")
    print(classifier.predict(X_test))
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    # Fit the classifier
    classifier.fit(X_train, y_train)
    
    # Save the classifier
    dump(classifier, 'AdaBoostSVC.xml')
    print("Ada Testing Predictions")
    print(classifier.predict(X_test))
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall

# =====================================================================


def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, precision, recall)
    
    All the elements in results will be plotted.
    '''

    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + URL)

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================


if __name__ == '__main__':

    #X_train = []
    #Y_train = []

    #p_size = 0
    #n_size = 0
    #dir_p = r"C:/Users/Niraj93/source/repos/Melanoma_Detection/Melanoma_Detection/data/data/data_positive/"
    #dir_n = r"C:/Users/Niraj93/source/repos/Melanoma_Detection/Melanoma_Detection/data/data/data_negative/"
    #for f in os.listdir(dir_p):
    #   if f.endswith('.jpg'):
    #       #files.append(dir_p+f)
    #       img = cv.imread(dir_p+f)
    #       img = cv.resize(img,(640,480))
    #       if img is None:
    #            raise Exception("No Image Loaded !")
    #       X_train.append(np.array(feature_extract(img,Segment(img))))
    #       Y_train.append(np.array(1))
    #       #TrainingSamples.append(feature_extract(img,Segment(img)))
    #       #label.append(1)

    #for f in os.listdir(dir_n):
    #   if f.endswith('.jpg'):
    #       #files.append(dir_n+f)
    #       img = cv.imread(dir_n+f)
    #       img = cv.resize(img,(640,480))
    #       if img is None:
    #            raise Exception("No Image Loaded !")
    #       X_train.append(np.array(feature_extract(img,Segment(img))))
    #       Y_train.append(np.array(-1))
    #       #TrainingSamples.append(feature_extract(img,Segment(img)))
    #       #label.append(-1)
            

   
    #cv.imshow('name',mask)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


    #THIS SECTION CAN BE REMOVED ONCE SVMS ARE TRAINED============
    # Process data into feature and label arrays
    """X_train, X_test, y_train, y_test = get_features_and_labels()

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    plot(results)"""
    #END REMOVED SECTION=========================================

    #Rahul, this is where the loading and classification is done
    #Classify is really straight forward and just needs an image and will provide a float as result
    #not sure how the GUI will give the image but you can just pass that to classify
    #you can have the classify function called on button press, but you'll need to
    #find a way to keep the SVM's loaded.
    
    from joblib import load
    linear_svc =  load('LinearSVC.xml')
    nu_svc = load('NuSVC.xml')
    ada_boost =  load('AdaBoostSVC.xml')

    img = cv.imread("G:/Melanoma Detection/data/data_positive/ISIC_0000144.jpg")
    if img is None:
        raise Exception("No Image Loaded !")
        print("No Image Loaded");
    
    print("The Linear classification is: ")
    print(classify(img, linear_svc))

    print("The NuSVC classification is: ")
    print(classify(img, nu_svc))

    print("The AdaBoost classification is: ")
    print(classify(img, ada_boost))
    

