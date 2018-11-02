
# coding: utf-8

# #  Current dev:
# * ~~Fix so PCA plots are not shown if not needed DONE~~
# * ~~Fix so an entire scree plot is shown if true and only 2 components are run if false DONE~~
# * ~~Remove the extra df in the clustering step~~
# * ~~Add scaling with z-score~~
# * Plot box plots and ~~ouput summary~~
# * ~~Plot correlations~~
# * Deal with extreme values
# * Can we create a facet grid style
# * Change the rotations to show in the PCA
# * Add extra clustering methods - dbscan etc.
# * Add error handling checking to make sure inputs are correct
# * Check summary statistics of the data
# * Output segmentation size
# * Wrap in compiler function
# * circle the current number of components
# * suggest the best number of components
# * suggest the best number of clusters
# * run more than one cluster solution at once maybe?
# * pd_display
# * heatmap everything
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os
from IPython.display import display, Image
import seaborn as sns
import matplotlib.pyplot as plt


get_ipython().magic('matplotlib inline')


# In[5]:


#Parameters for the 

data_loc = '/nfs/science/globalscience/tomev/store_clustering/store_cluster_data.csv'

#clustering universe is the overlay metrics - not currently being used
clustering_universe = 'format'

#define feature labels of input data, to be used later to subset features to cluster on
feature_group_labels = ['label','format','postcode','region','category','category','category','category','category',
           'category','category','category','category','category','category','category','category',
           'category','category','category','category','category','category','category','category',
           'category','category','category','category','category','category','category','category',
           'category','category','category','category','category','category','category','category',
           'category','category','category','category','size','size']

#feature groups to build clusters on
feature_groups_to_cluster = ['category','size']

#normalisation
Norm=True
Method='min max'

#pre-processing
PP_pca = True
pp_pca_num = 2




#range of number of clusters to be explored (e.g. 2 to 7 is 6 cluster solutions from a 2 cluster solution
#to a 7 cluster solution)
min = 2
max = 7 
cluster_range = range(min,max+1)

#define the cluster solution methodology
CS_method = 'kmeans'


#Colours for the plot
#b: blue
#g: green
#r: red
#c: cyan
#m: magenta
#y: yellow
#k: black
colours = ['b','g','r','c','m','y','k']

print ('Parameters set')


# In[6]:


#Loop through the clustering and calculate the cluster quality

#Import data
def import_data(filename,file_type='csv'):
    '''Import data function:
    filename: location and filename of data to import (e.g. '/nfs/science/globalscience/tomev/store_clustering/store_cluster_data.csv')
    file_type: type of text file to import either csv or txt (default = csv)'''
    print ('Importing data from: ' + filename)
    if file_type == 'csv':
        data=pd.read_csv(filename)
    elif file_type == 'txt':
         data=pd.read_table(filename)
    else:
        raise NameError('Data type not recognised')
    print ('Import complete.\nRows: ' + str(len(data)) + '\nCols: ' + str(len(data.columns)))
    return data

def data_summary(data=None):
    '''Display a summary of the input data (default=cluster_data)'''
    if data==None:
        data=cluster_data
    return display(data.describe())
        
#subset the data
def subset(data=None, Feature_groups_to_cluster=None, Feature_group_labels=None):
    '''Subsets input data to remove features that will not be used in the cluster:
    data: default = cluster_data
    Feature_groups_to_cluster: the features you would like to cluster on
    Feature_group_labels: labels of the entire feature set'''
    print ('Subsetting data')
    if Feature_group_labels == None:
        Feature_group_labels=feature_group_labels
    if Feature_groups_to_cluster == None:
        Feature_groups_to_cluster=feature_groups_to_cluster
    if data == None:
        data=cluster_data
    col_names = []
    for i in xrange(len(Feature_group_labels)):
        if Feature_group_labels[i] in Feature_groups_to_cluster:
            col_names.append(data.columns[i])
    data= data[col_names]
    num_cols = len(data.columns)
    return data, num_cols

#clean nas
def clean_nas(data=None,Value=0):
    '''Clean the missing values of input data:
    data: default = cluster_data
    Value: value to exchange missing values for (default=0)'''
    print ('Changing missing values to: ' + str(Value))
    if data==None:
        data = cluster_data
    data.fillna(value=Value, inplace=True )
    


#normalise data
def normalise_data(data=None,method=Method,norm=Norm):
    '''Normalise input data:
    data: default = cluster_data
    method: normalisation methodology (options=min max and z-score standardisation
    norm: True or False, if False this function will skip without normalisation)'''
    if norm==False:
        pass
    elif norm==True:
        print ('Normalising the data using method: ' + method)
        if data==None:
            data=cluster_data
        if method == 'min max':
            x = data.values #returns a numpy array
            headers=data.columns
            Norm = preprocessing.MinMaxScaler()
            x_norm = Norm.fit_transform(x)
            data = pd.DataFrame(x_norm)
            data.columns=headers
            return data
        if method == 'standard':
            x = data.values #returns a numpy array
            headers=data.columns
            Norm = preprocessing.StandardScaler()
            x_norm = Norm.fit_transform(data)
            data = pd.DataFrame(x_norm)
            data.columns=headers
            return data
    else:
        raise ValueError("norm must be True or False")

        
def plot_corr(data=None):
    '''Correlation matrix of the input data:
    data: default = cluster_data'''
    print('\n---------------------------------------\nPlot of correlation matrix\n')
    if data==None:
        data=cluster_data
    corr = data.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.show()
        
#PCA
def PCA_func(data=None,num_components=None,pre_process=None):
    ''''''
    if data==None:
        data=cluster_data
    if pre_process==None:
        pre_process=PP_pca
    if pre_process==True:
        if num_components == None:
            num_components=pp_pca_num
            print('\n---------------------------------------\nRunning PCA with ' +str(num_components) + ' components to be input into the clustering' )
        if num_components<4:
            pca =  PCA(n_components=len(data.columns))
            pca_components =pca.fit_transform(data)
            for i in range(4):
                data['PC_' + str(i)]=pca_components[:,i]
            return data, pca
        else:
            pca =  PCA(n_components=len(data.columns))
            pca_components =pca.fit_transform(data)
            for i in range(num_components):
                data['PC_' + str(i)]=pca_components[:,i]
            return data, pca
    elif pre_process==False:
        print('\n---------------------------------------\nRunning PCA output not to be used in model' )
        num_components=pp_pca_num
        pca =  PCA(len(cluster_data.columns))
        pca_output =pca.fit_transform(data)
        for i in xrange(4):
            data['PC_' + str(i)]=pca_output[:,i]
        return data, pca

#Scree plot of PCA
def pca_scree_plot(pca_model=None,pre_process=None):
    if pre_process==None:
        pre_process=PP_pca
    if pre_process==True:
        if pca_model==None:
            pca_model=pca
        #create plot
        print('\n---------------------------------------\nScree plot of explained variance with ' )
        x = np.arange(0,len(pca.explained_variance_))
        plt.scatter(x,pca.explained_variance_,color='k')
        plt.plot(x,pca.explained_variance_)
        # Add legend and details
        plt.title('Scree plot of topic probabilities PCA', fontsize=16)
        plt.xlabel('Dimensions')
        plt.ylabel('Explained variance')
        plt.show()
    elif pre_process==False:
        pass

#Plot of the components
def component_plot(data=None,n_components=None,pre_process=None):
    if pre_process==None:
        pre_process=PP_pca
    if pre_process==True:
        print('\n---------------------------------------\nPlot of principle axes' )
        if data==None:
            data=cluster_data        
        if n_components==None:
            n_components=pp_pca_num
        for i in xrange(n_components-1):
            x = data['PC_'+str(i)]
            y = data['PC_'+str(i+1)]
            plt.title('Plot of the '+str(i+1)+' & '+str(i+2) + ' component', fontsize=16)
            plt.xlabel('Component '+str(i))
            plt.ylabel('Component '+str(i+1))
            plt.scatter(x,y,color='k')
            plt.show()

        for i in xrange(n_components-1):
            x = data['PC_'+str(0)]
            y = data['PC_'+str(i)]
            if i==0:
                pass
            else:
                plt.title('Plot of the '+str(1)+' & '+str(i+2) + ' component', fontsize=16)
                plt.xlabel('Component '+str(0))
                plt.ylabel('Component '+str(i+1))
                plt.scatter(x,y,color='k')
                plt.show()            
    elif pre_process==False:
        pass

    
#CHANGE H CLUST TO BE SCIPY SO WE CAN PLOT THE DENDROGRAM
def cluster(data=None,cluster_sol=None,pp_pca=None):
    if data==None:
        data=cluster_data
    if cluster_sol==None:
        cluster_sol=CS_method
    if pp_pca==None:
        pp_pca=PP_pca
    print('\n---------------------------------------\nRunning ' +cluster_sol+' cluster with range:'+str(cluster_range) + '\n')
    if cluster_sol == 'kmeans':
        if pp_pca == False:
            data=data.iloc[:,:num_columns]
            inertia=[]
            for i in cluster_range:
            #model data
                model=KMeans(n_clusters=i,init='k-means++',n_jobs=1)
                output_model = model.fit(data)
                cluster_data['kmeans_' + str(i)] = model.labels_
                inertia.append([i,model.inertia_])
            inertia = np.asarray(inertia).T
        elif pp_pca == True:
            data=data.iloc[:,num_columns:]
            inertia=[]
            for i in cluster_range:
            #model data
                model=KMeans(n_clusters=i,init='k-means++',n_jobs=1)
                output_model = model.fit(data)
                cluster_data['kmeans_' + str(i)] = model.labels_
                inertia.append([i,model.inertia_])
            inertia = np.asarray(inertia).T
        fig2 = plt.figure( figsize = (7, 5) )
        plt.scatter(inertia[0],inertia[1],color='k')
        plt.plot(inertia[0],inertia[1])
        plt.title('Plot of inertia decline with number of clusters (k)\n', fontsize=16)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.show()
    elif cluster_sol == 'hierarchical':
        if PP_pca == False:
            data=data.iloc[:,:num_columns]
            for i in cluster_range:
                model = AgglomerativeClustering(n_clusters=i, linkage='ward')
                output_model = model.fit(data)
                cluster_data['hierarchical_' + str(i)] = model.labels_
        if PP_pca == True:
            data=data.iloc[:,num_columns:]
            for i in cluster_range:
                model = AgglomerativeClustering(n_clusters=i, linkage='ward')
                output_model = model.fit(data)
                cluster_data['hierarchical_' + str(i)] = model.labels_

#create the plot of the clusters
def plot_clusters(data=None,Cluster_range=None,colour_range=None,label_column_name=None):
    if colour_range==None:
        colour_range=colours
    if data==None:
        data=cluster_data
    if Cluster_range==None:
        Cluster_range=cluster_range
    if label_column_name==None:
        label_column_name = CS_method
        
    for i in range(len(Cluster_range)):
        for j in range(Cluster_range[i]):
            x = data['PC_0'][data[label_column_name + '_' + str(Cluster_range[i])]==j]
            y = data['PC_1'][data[label_column_name + '_' + str(Cluster_range[i])]==j]
            plt.scatter(x,y,color=colour_range[j-1],s=1)
            plt.title('Plot of ' + str(Cluster_range[i]) + ' cluster solution\nPC 1 & 2\n',fontsize=16)
            plt.xlabel('Principle comp0nent 1')
            plt.ylabel('Principle component 2') 
        plt.show()
        for j in range(Cluster_range[i]):
            x = data['PC_0'][data[label_column_name + '_' + str(Cluster_range[i])]==j]
            y = data['PC_2'][data[label_column_name + '_' + str(Cluster_range[i])]==j]
            plt.scatter(x,y,color=colour_range[j-1],s=1)
            plt.title('Plot of ' + str(Cluster_range[i]) + ' cluster solution\nPC 1 & 3\n',fontsize=16)
            plt.xlabel('Principle comp0nent 1')
            plt.ylabel('Principle component 3')
        plt.show()
        for j in range(Cluster_range[i]):
            x = data['PC_0'][data[label_column_name + '_' + str(Cluster_range[i])]==j]
            y = data['PC_3'][data[label_column_name + '_' + str(Cluster_range[i])]==j]
            plt.scatter(x,y,color=colour_range[j-1],s=1)
            plt.title('Plot of ' + str(Cluster_range[i]) + ' cluster solution PC\nPC 1 & 4\n',fontsize=16)
            plt.xlabel('Principle component 1')
            plt.ylabel('Principle component 4') 
        plt.show()

def cluster_sizes(data=None,label_column_name=None,Range=None):
    if data == None:
        data=cluster_data
    if label_column_name==None:
        label_column_name=CS_method
    if Range==None:
        Range=cluster_range
    print('\n---------------------------------------\nTable of cluster sizes\n')
    for i in Range:
        display('Table of cluster sizes where number of clusters is '+str(i))
        display(pd.DataFrame(cluster_data.groupby(label_column_name+'_'+str(i))[label_column_name+'_'+str(i)].count().values,
                             columns=[label_column_name+'_'+str(i)]))

def profile_cluster(data=None,cols_profile=None,cluster_col=None):
    if data<>None and cols_profile==None:
        raise ValueError('Columns to be profiled must be specified as a list')
    elif data==None and cols_profile==None:
        data=cluster_data
        cols_profile=data.iloc[range(num_columns),:].columns.values
    display(data.groupby(cluster_col)[data.iloc[:,range(num_columns)].columns.values].mean())
    


# In[7]:


#Running of to investigate the clusters
cluster_data = import_data(data_loc)
data_summary()
cluster_data, num_columns = subset()
clean_nas()
cluster_data = normalise_data()
data_summary()
plot_corr()
cluster_data, pca = PCA_func()
pca_scree_plot()
component_plot()
cluster()
plot_clusters()
cluster_sizes()


# In[8]:


profile_cluster(cluster_col='kmeans_4')

