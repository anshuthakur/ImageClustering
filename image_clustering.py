import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_rn50
from shutil import copyfile
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class ClusteringImages:
    """
    The class Clustering Images uses a pretrained ResNet50 model and Kmeans clustering to group the images into different
    clusters. 
    """

    def __init__(self):
        pass
  
    def get_vectors(self, image_path, model, preprocess_input):
        """
        Gets a list of features of different images.
        """
        model.layers[0].trainable = False
        image_feature_list = []
        for f in os.listdir(image_path):
            fullpath = os.path.join(image_path, f)
            im = cv2.imread(fullpath)

            #resize image
            im = cv2.resize(im, (224, 224))
            im_data = image.img_to_array(im)

            im_data = np.expand_dims(im_data,axis = 0)
            im_data = preprocess_input(im_data)

            features = model.predict(im_data)
            features_toarray = np.array(features)

            image_feature_list.append(features_toarray.flatten())
        return image_feature_list
  
    def _get_clusters(self, feature_array, n_clusters = 6):
        """
        Uses scikit-learn's Kmeans algorithm for clustering the input into different groups.
        """
        kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(feature_array)
        return kmeans
  
    def _cluster_images(self, model_name='ResNet50'):
        
        image_path = self.image_path
        model_dict = {"ResNet50": [ResNet50(include_top=False, weights = "imagenet", pooling = "max"), preprocess_input_rn50]
                  }
        model, preprocess_input = model_dict[model_name]

        feature_list = self.get_vectors(image_path, model, preprocess_input)

        kmeans_model = self._get_clusters(feature_list, self.n_clusters)
        return kmeans_model
    
    def get_clusters(self, image_path,model="ResNet50", n_clusters = 6, save_images = False):
        self.image_path = image_path
        self.n_clusters = n_clusters
        self.kmeans_model = self._cluster_images(model)
        
        # to save the images into different folders based on clusters.
        if save_images:
            self.group_images()
        
        self.display_images()
    
    def group_images(self):
        """
        Adds images into different folders based on clusters.
        """
        labels = self.kmeans_model.labels_
        image_list = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
        
        for n in range(self.n_clusters):
            dest_path = os.path.join(os.getcwd(), str(n))
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            for i in range(len(image_list)):
                if labels[i] == n:
                    dest_filepath = os.path.join(dest_path, os.path.basename(image_list[i]))
                    copyfile(image_list[i], dest_filepath)
                    
    
    def display_images(self):
        """
        Displays the grouped-images label wise.
        """
        labels = self.kmeans_model.labels_
        
        image_list = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
        
        for n in range(6):
            print(f"Label - {n}")
            for i in range(len(image_list)):
                if labels[i] == n:
                    im = cv2.imread(image_list[i])
                    im_new = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    plt.imshow(im_new)
                    plt.show()
            print("*"*120)        

if __name__ =="__main__":
    
    ci = ClusteringImages()
    image_path = "pics/"

    ci.get_clusters(image_path = image_path, n_clusters = 6, save_images = True)
