import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import io
from skimage.filters import threshold_otsu 
import tensorflow as tf
import pandas as pd
import numpy as np
from time import time
import csv
import keras
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
from colorama import Fore, Style

tf.disable_v2_behavior() 

# paths to images
genuine_image_paths = "real"
forged_image_paths = "forged"


def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg

def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
#     img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg

def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img) #rgb to grey
    if display:
        plt.imshow(grey, cmap = matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey) #grey to binary
    if display:
        plt.imshow(binimg, cmap = matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg==1)
    # Now we will make a bounding box with the boundary as the position of pixels on extreme.
    # Thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap = matplotlib.cm.Greys_r)
        plt.show()
    return signimg

def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                a = a+1
    total = img.shape[0] * img.shape[1]
    return a/total

def Centroid(img):
    numOfWhites = 0
    a = np.array([0,0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                b = np.array([row,col])
                a = np.add(a,b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a/numOfWhites
    centroid = centroid/rowcols
    return centroid[0], centroid[1]

def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity


def SkewKurtosis(img):
    h,w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    #calculate projections along the x and y axes
    xp = np.sum(img,axis=0)
    yp = np.sum(img,axis=1)
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))
    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3

    return (skewx , skewy), (kurtx, kurty)



def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal

def create_placeholder_image(path):
    print(f"Creating placeholder image at: {path}")
    img = np.zeros((100, 100, 3), dtype=np.uint8) 
    os.makedirs(os.path.dirname(path), exist_ok=True)  
    mpimg.imsave(path, img)

def getCSVFeatures(path, img=None, display=False):
    if not os.path.exists(path):
        print(f"File not found: {path}. Creating placeholder.")
        create_placeholder_image(path)

    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features

def count_training_csv():
    training_folder = 'Features/Training'
    if os.path.exists(training_folder):
        csv_files = [f for f in os.listdir(training_folder) if f.startswith('training_') and f.endswith('.csv')]
        csv_ids = [f.split('_')[1].split('.')[0] for f in csv_files]
        return sorted(csv_ids, key=int)  # Return sorted IDs as a list
    else:
        print(f"Training folder '{training_folder}' does not exist.")
        return []

def makeCSV():
    if not(os.path.exists('Features')):
        os.mkdir('Features')
        print('New folder "Features" created')
    if not(os.path.exists('Features/Training')):
        os.mkdir('Features/Training')
        print('New folder "Features/Training" created')
    if not(os.path.exists('Features/Testing')):
        os.mkdir('Features/Testing')
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    training_csv_ids = count_training_csv()
    # print("------------total csv----------------", training_csv_ids)
    for person in training_csv_ids:
        per = ('00'+str(person))[-3:]
        print(Fore.BLUE + 'DATA SET LOADED ! DONE -', per + Style.RESET_ALL)
        
        with open(f'Features/Training/training_{person}.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(0,3):
                source = os.path.join(gpath, f"{person}{person}_00{i}.png")
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(0,3):
                source = os.path.join(fpath, f"021{person}_00{i}.png")
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
        
        with open('Features/Testing/testing_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(3, 5):
                source = os.path.join(gpath, f"{person}{person}_00{i}.png")
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(3,5):
                source = os.path.join(fpath, f"021{person}_00{i}.png")
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
            



def testing(path):
    feature = getCSVFeatures(path)
    if not(os.path.exists('TestFeatures')):
        os.mkdir('TestFeatures')
    with open('TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')
    return feature

    
makeCSV()

def ensure_csv_exists(person_id):
    """Ensure the training and testing CSV files exist for the given person ID."""
    training_file = f'Features/Training/training_{person_id}.csv'
    testing_file = f'Features/Testing/testing_{person_id}.csv'

    # Check and create the training file if it doesn't exist
    if not os.path.exists(training_file):
        print(f'Training file for person ID {person_id} not found. Creating a new file.')
        with open(training_file, 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')

    # Check and create the testing file if it doesn't exist
    if not os.path.exists(testing_file):
        print(f'Testing file for person ID {person_id} not found. Creating a new file.')
        with open(testing_file, 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')

def load_person_features(person_id):
    csv_path = f'Features/Training/training_{person_id}.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No training data found for person ID: {person_id}")

    features_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            features = np.array([float(row[key]) for key in row if key != 'output'])
            features_list.append(features)
    return np.array(features_list)

def compare_signature(person_id, signature_path):
    """Compare the provided signature with the stored signatures for a given person."""
    print(f"Comparing provided signature with person ID: {person_id}")
    try:
        test_features = getCSVFeatures(signature_path)
        print("Extracted features from provided signature:", test_features)

        # Load stored features for the person
        stored_features = load_person_features(person_id)
        print(f"Loaded {len(stored_features)} features for person ID: {person_id}")

        # Compare features using Euclidean distance
        distances = np.linalg.norm(stored_features - test_features, axis=1)
        print("Distances to stored features:", distances)

        # Set a threshold for signature match
        threshold = 5.0  # Adjust based on your data
        if np.any(distances < threshold):
            print("Signature matches with the stored data!")
            return "Signature Match"
        else:
            print("Invalid signature!")
            return "Invalid Signature"
    except FileNotFoundError as e:
        print(e)
        return str(e)


n_input = 9
train_person_id = "001"
# test_image_path1 = input("Enter path of signature image 1 : ")
# test_image_path2 = input("Enter path of signature image 2 : ")
# ensure_csv_exists(train_person_id)
train_path = 'Features/Training/training_'+train_person_id+'.csv'
# test_image_path1 = input("Enter path of signature image 1: ")
# test_image_path2 = input("Enter path of signature image 2: ")

# Extract features using the testing() function
# image_1 = testing(test_image_path1)
# image_2 = testing(test_image_path2)

# data_array_1 = np.array(image_1, dtype=np.float64)
# data_array_2 = np.array(image_2, dtype=np.float64)

# # Print extracted features
# print("------------Image 1 Features-----", data_array_1)
# print("------------Image 2 Features-----", data_array_2)

# # Normalize feature vectors
# data_array_1 /= np.linalg.norm(data_array_1)
# data_array_2 /= np.linalg.norm(data_array_2)

# # Calculate Euclidean distance and Cosine similarity
# euclidean_distance = np.linalg.norm(data_array_1 - data_array_2)
# cosine_similarity = np.dot(data_array_1, data_array_2)

# # Print calculated metrics
# print("Euclidean Distance:", euclidean_distance)
# print("Cosine Similarity:", cosine_similarity)

# # Define thresholds for matching
# euclidean_threshold = 0.7  # Higher threshold for allowing small differences
# cosine_threshold = 0.75      # Higher threshold for similarity

# # Determine if the images match
# if euclidean_distance < euclidean_threshold and cosine_similarity > cosine_threshold:
#     print("Images match.")
# else:
#     print("Images do not match.")

# result = compare_signature(train_person_id, test_image_path1)
# print( 'SIGNATURE STATUS ===> ', result)
test_path = 'TestFeatures/testcsv.csv'



def readCSV(train_path, test_path, type2=False):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(correct,2)      # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not(type2):
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = kearas.utils.to_categorical(correct,2)      # Converting to one hot
    if not(type2):
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input

ops.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 7 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
n_hidden_3 = 30 # 3rd layer
n_classes = 2 # no. of classes (genuine or forged)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# For accuracies
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

def evaluate(train_path, test_path, type2=False):   
    if not(type2):
        train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(train_path, test_path, type2)
    ans = 'Random'
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            if cost<0.0001:
                break
#             # Display logs per epoch step
#             if epoch % 999 == 0:
#                 print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
#         print("Optimization Finished!")
        
        # Finding accuracies
        accuracy1 =  accuracy.eval({X: train_input, Y: corr_train})
#         print("Accuracy for train:", accuracy1)
#         print("Accuracy for test:", accuracy2)
        if type2 is False:
            accuracy2 =  accuracy.eval({X: test_input, Y: corr_test})
            return accuracy1, accuracy2
        else:
            prediction = pred.eval({X: test_input})
            if prediction[0][1]>prediction[0][0]:
                print('Genuine Image')
                return True
            else:
                print('Forged Image')
                return False


def trainAndTest(rate=0.001, epochs=1700, neurons=7, display=False):    
    start = time()

    # Parameters
    global training_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs

    # Network Parameters
    n_hidden_1 = neurons # 1st layer number of neurons
    n_hidden_2 = 7 # 2nd layer number of neurons
    n_hidden_3 = 30 # 3rd layer

    train_avg, test_avg = 0, 0
    n = 10
    for i in range(1,n+1):
        if display:
            print("Running for Person id",i)
        temp = ('0'+str(i))[-2:]
        train_score, test_score = evaluate(train_path.replace('01',temp), test_path.replace('01',temp))
        train_avg += train_score
        test_avg += test_score
    if display:
#         print("Number of neurons in Hidden layer-", n_hidden_1)
        print("Training average-", train_avg/n)
        print("Testing average-", test_avg/n)
        print("Time taken-", time()-start)
    return train_avg/n, test_avg/n, (time()-start)/n


evaluate(train_path, test_path, type2=True)

