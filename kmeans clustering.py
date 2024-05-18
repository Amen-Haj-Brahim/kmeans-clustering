# Note
### This entire thing can also be done using the skimage.segmentation.slic() method
# good old importing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
def kmeans(img):
    print("-----------------------------------------------------------------------------------------------------")
    plt.imshow(img)
    plt.show()
    
    # image resolution (the 3 values are rgb channels)
    imgres=img.shape
    print("image shape before : ",imgres)
    
    # falttening the image (i'm not sure that's the right word ?) to a 1D array with rgb channels
    img=img.reshape((-1,3))
    print("image shape after:",img.shape)
    
    # Segmentation
    ## Trying 3 Clusters
    km=KMeans(n_clusters=3,verbose=0,n_init=10)
    km.fit(img)
    
    # printing the set to show unique clusters labels only
    labels=km.labels_ 
    print("unqiue labels : ",set(labels),labels.shape)
    
    # getting cluster center points
    centers=np.array(km.cluster_centers_,dtype='uint8')
    print("clusters centers : ",centers)
    
    # each cluster gets pixels that are associated with the clusters center
    segmented_values=centers[labels]
    segmented_img=segmented_values.reshape((imgres))
    
    # result
    plt.imshow(segmented_img)
    plt.show()
    
    # Comparing clustering from 1 to 10 clusters
    # same thing but looping through the number of clusters and comparing the result
    imgs=[]
    for i in range(1,10):
        
        kmIMG=KMeans(n_clusters=i,verbose=0,n_init=10)
        kmIMG.fit(img)
        
        labels=kmIMG.labels_
        centers=np.array(kmIMG.cluster_centers_,dtype='uint8')
        
        segmented_values=centers[labels]
        segmented_img=segmented_values.reshape((imgres))
        
        imgs.append(segmented_img)
    
    imgs=np.array(imgs)
    inertia_values=[]
    
    plt.figure(figsize=(10,5))
    
    for i in range(1, len(imgs) + 1):
        
        plt.subplot(2, 5, i)
        plt.imshow(imgs[i - 1])
        if i==1:
            plt.xlabel(str(i) + " cluster")    
        else:
            plt.xlabel(str(i) + " clusters")
        plt.yticks([])
        plt.xticks([])
    plt.show()
    
    # Evaluation
    ## Getting the Inertia for clusters from 1 to 10
    inertia_values=[]
    for i in range(1,11):
        km=KMeans(n_clusters=i,n_init='auto')
        km.fit_predict(img)
        inertia_values.append(km.inertia_)
    print("inertia of 1 to 10 clusters : ",inertia_values)
    
    plt.plot(inertia_values)
    plt.xlabel("n clusters")
    plt.ylabel("inertia")
    plt.xticks([i for i in range(0,10)])
    plt.show()
    ## This part takes alot of time (i interrupted it at 70 mins because i'm not waiting any longer whataver) to execute so ignore it if you want
    """silhouette_values=[]
    for i in range(2,11):
        km=KMeans(n_clusters=i)
        km.fit_predict(img)
        silhouette_values.append(silhouette_score(img,km.fit_predict(img)))
    silhouette_values"""
    """plt.plot(silhouette_values)
    plt.xlabel("n clusters")
    plt.ylabel("silhouette score")
    plt.xticks([i for i in range(0,10)])"""
    
# loading the image and displaying it
imgs=[plt.imread("pic1.jpg"),plt.imread("pic2.jpg")]
for img in imgs:
    kmeans(img)