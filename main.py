import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

FOLDER = "./Dataset/"
FILES = os.listdir(FOLDER)
TEST_DIR = "./Testset/"

def load_images_train_and_test(TEST):
    test=np.asarray(Image.open(TEST)).flatten()
    train=[]
    for name in FILES:
        train.append(np.asarray(Image.open(FOLDER + name)).flatten())
    train= np.array(train)
    return test,train
   
def normalize(test,train):
    normalized_test = test - np.mean(train, 0)
    normalized_train = train - np.mean(train, 0)
    return normalized_test,normalized_train


def svd_function(images):
    u, s, v =  np.linalg.svd(images, full_matrices=False)
    return u,s,v


def project_and_calculate_weights(img,u):
    return np.multiply(img, u)


def predict(test,train):
    
    sub = np.subtract(train[:,0], test)
    least_error = np.linalg.norm(sub)
    index = 0
    for i in range(train[0].size) :
        sub = np.subtract(train[:,i], test)
        norm = np.linalg.norm(sub)
        if(norm < least_error):
            least_error = norm
            index = i
    return index


def plot_face(tested,predicted):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(tested, cmap = "gray")
    ax2.imshow(predicted, cmap = "gray")
    plt.show()
    # pass

if __name__ == "__main__":
    true_predicts=0
    all_predicts=0
    for TEST_FILE in os.listdir(TEST_DIR):
        # Loading train and test
        test,train=load_images_train_and_test(TEST_DIR+TEST_FILE)

        # Normalizing train and test
        test,train=normalize(test,train)
        test=test.T
        train=train.T
        test = np.reshape(test, (test.size, 1))

        # Singular value decomposition
        u,s,v=svd_function(train)

        # Weigth for test
        w_test=project_and_calculate_weights(test,u)
        w_test=np.array(w_test, dtype='int8').flatten()

        # Weights for train set
        w_train=[]
        for i in range(train.shape[1]):
            w_i=project_and_calculate_weights(np.reshape(train[:, i], (train[:, i].size, 1)),u)
            w_i=np.array(w_i, dtype='int8').flatten()
            w_train.append(w_i)
        w_train=np.array(w_train).T

        # Predict 
        index_of_most_similar_face=predict(w_test,w_train)

        # Showing results
        print("Test : "+TEST_FILE)
        print(f"The predicted face is: {FILES[index_of_most_similar_face]}")
        print("\n***************************\n")

        # Calculating Accuracy
        all_predicts+=1
        if FILES[index_of_most_similar_face].split("-")[0]==TEST_FILE.split("-")[0]:
            true_predicts+=1
            # Plotting correct predictions 
            plot_face(Image.open(TEST_DIR+TEST_FILE),Image.open(FOLDER+FILES[index_of_most_similar_face]))
        # else:
        #     # Plotting wrong predictions
        #     plot_face(Image.open(TEST_DIR+TEST_FILE),Image.open(FOLDER+FILES[index_of_most_similar_face]))

    # Showing Accuracy
    accuracy=true_predicts/all_predicts
    print(f'Accuracy : {"{:.2f}".format(accuracy*100)} %')
        
    