from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def kmeans(sampled_scaler, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)

    cluster = model.fit_predict(sampled_scaler)
    return cluster


def run_elbow():
    n_clusters=10
    cost=[]
    for i in range(1,n_clusters):
        kmean= KMeans(i).fit(df_copy)
        cost.append(kmean.inertia_)
    
    plt.plot(cost, 'bx-')

 
def pca_model(data, columns=['PCA1','PCA2']):
    pca=PCA(n_components=2)
     
    reduced_X=pd.DataFrame(data=pca.fit_transform(data),columns=columns)
 
 
    display(reduced_X.head())
    return reduced_X 
 