#!/usr/bin/env python
# coding: utf-8

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162324.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162324.png)

# # Denetimsiz Öğrenme

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141342.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141342.png)

# # K-Means

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141543.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141543.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141641.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141641.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141951.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20141951.png)

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[13]:


data = pd.read_csv(r"C:\data_set\USArrests.csv", index_col=0 )
data.head()


# In[14]:


data.isnull().sum()


# In[15]:


data.info()


# In[16]:


data.describe().T


# In[17]:


data.hist(figsize=(10,10));


# In[18]:


kmeans= KMeans(n_clusters= 4)


# In[19]:


kmeans


# In[20]:


k_fit= kmeans.fit(data)


# In[21]:


k_fit.n_clusters


# In[22]:


k_fit.cluster_centers_


# In[23]:


k_fit.labels_


# In[24]:


# kümelerin görselleştirilmesi


# In[25]:


k_means= KMeans(n_clusters= 2).fit(data)


# In[26]:


kumeler= k_means.labels_


# In[27]:


kumeler


# In[30]:


plt.scatter(data.iloc[:,0],data.iloc[:,1],c = kumeler, s=50, cmap= "viridis");


# In[31]:


merkezler= k_means.cluster_centers_


# In[32]:


merkezler


# In[36]:


plt.scatter(data.iloc[:,0],data.iloc[:,1],c = kumeler, s=50, cmap= "viridis");
plt.scatter(merkezler[:,0],merkezler[:,1], c= "black", s=200, alpha=0.5);

Elbow Yöntemi
# In[44]:


# Bellek sızıntısı uyarısını bastırmak için çözüm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[45]:


ssd= []

K= range(1,30)

for k in K:
    kmeans = KMeans(n_clusters= k).fit(data)
    ssd.append(kmeans.inertia_)


# In[46]:


plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")


# In[48]:


get_ipython().system('pip install yellowbrick')


# In[49]:


from yellowbrick.cluster import KElbowVisualizer


# In[52]:


kmeans= KMeans()
visu= KElbowVisualizer(kmeans, k=(2,20))
visu.fit(data)
visu.poof()


# In[53]:


kmeans= KMeans(n_clusters= 4).fit(data)


# In[54]:


kmeans


# In[55]:


kumeler = kmeans.labels_


# In[57]:


pd.DataFrame({"Eyaletler":data.index, "Kumeler": kumeler})


# In[58]:


data["kume_no"]= kumeler


# In[59]:


data


# # Hiyerarşik Kümeleme

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20155948.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20155948.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20160009.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20160009.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20160242.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20160242.png)

# In[60]:


from scipy.cluster.hierarchy import linkage


# In[67]:


hc_complete= linkage(data, "complete")
hc_average= linkage(data, "average")


# In[62]:


from scipy.cluster.hierarchy import dendrogram


# In[68]:


plt.figure(figsize=(15,10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
          leaf_font_size= 10);


# In[69]:


plt.figure(figsize=(15,10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
       truncate_mode = "lastp",
       p=10,
       show_contracted= True,
      leaf_font_size= 10);


# In[70]:


plt.figure(figsize=(15,10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
          leaf_font_size= 10);


# In[71]:


plt.figure(figsize=(15,10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode = "lastp",
           p=10,
           show_contracted= True,
          leaf_font_size= 10);


# # Temel Bileşen Analizi (PCA)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162057.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162057.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162159.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162159.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162224.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-22%20162224.png)

# In[78]:


data = pd.read_csv(r"C:\data_set\Hitters.csv")
data.dropna(inplace= True)
data = data._get_numeric_data()
data.head()


# In[74]:


from sklearn.preprocessing import StandardScaler


# In[80]:


data= StandardScaler().fit_transform(data) #standartlaşma yapıldı


# In[81]:


from sklearn.decomposition import PCA


# In[82]:


pca = PCA(n_components= 2)
pca_fit= pca.fit_transform(data)


# In[83]:


bilesen_df = pd.DataFrame(data = pca_fit, columns = ["birinci_bilesen","ikinci_bilesen"])


# In[84]:


bilesen_df


# In[87]:


pca.explained_variance_ratio_ # açıklandırma oranı


# In[88]:


pca.components_


# In[91]:


# optimum bilesen sayisi
pca = PCA().fit(data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen sayısı")
plt.ylabel("Kümülatif Varyans oranı");


# In[92]:


pca.explained_variance_ratio_


# In[93]:


# final
pca = PCA(n_components= 3)
pca_fit= pca.fit_transform(data)


# In[94]:


pca.explained_variance_ratio_


# In[ ]:




