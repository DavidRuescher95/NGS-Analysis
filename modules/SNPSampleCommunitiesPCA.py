import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
import networkx as nx
import networkx.algorithms.community as nx_comm
import string
# =============================================================================
# For dosage matrices
# =============================================================================
class SNPSampleCommunitiesPCA:
    '''
    A class for performing PCA-based clustering of dosage matrices.
    '''
    def __init__(self,dosageMatrix):
        '''
        Initialize the SampleCommunitiesPCA class.

        Parameters:
            - dosageMatrix: data frame, raw data generated from a vcf file.

        '''
        self.dosageMatrix = dosageMatrix.dropna(axis=0).T
        self.Samples = self.dosageMatrix.index.array
        self.Features = self.dosageMatrix.columns.array
        self.metaData = pd.DataFrame(data={"Cluster":np.nan},index=self.Samples)
    def scale(self,with_std=False):        
        '''
        Required scaling of the data using sklearns StandardScaler

        Parameters:
            - with_sd: True/False, gives user choice to center or scale the data

        Note: scaling is highly encouraged as it usually improves results.

        '''
        scaler = StandardScaler(with_std=with_std)        
        self.Scaled = scaler.fit_transform(self.dosageMatrix)        
        return self    
    def pca(self):
        '''
        Raises:
            - RuntimeError: Check, if scale were executed, as they are required.

        Returns:
            - Features: array of int, Indices of genes to use as input for PCA.
            - Genes: array of str, gene IDs of selected Features.
            - PCA_input: array, input for PCA.
            - PCA: sklearn object
        '''
        if self.Scaled is None:
            raise RuntimeError("Scaled data not found. Did you forget to run .scale() ?")
        self.PCA_input = self.Scaled
        pca = PCA(n_components=min(len(self.Samples),len(self.Features)))
        self.PCA = pca.fit(self.PCA_input)
        return self
    def louvain(self,minPCs=10,minPctVar=90,nNeighbors=10,resolution=1):
        '''
        Parameters:
            - minPCs: int, The minimum number of PCs to be used for graph generation
            - minPctVar: float, the minimum variance to be explained by the input
            - nNeighbors: int, the number of neighbors for kNN graph
            - resolution: int, resolution used by the louvain algorithm

        Raises:
            - RuntimeError: Check, if PCA was executed beforehand

        Returns:
            - Clusters: array, cluster information for each samples in metaData.index
            - metaData: dataframe, adds new column to existing metaData with cluster information
        '''
        if self.PCA is None:
            raise RuntimeError("PCA data not found. Did you forget to run .pca() ?")
        cumVar = self.PCA.explained_variance_ratio_.cumsum()
        pcs = max(
            len(cumVar[cumVar<(minPctVar/100)])+1,
            minPCs
             )
        self.PCs = pcs
        self.PCA_output = self.PCA.fit_transform(self.PCA_input)[:,[*range(pcs)]]
        G = kneighbors_graph(
            self.PCA_output,
            n_neighbors = nNeighbors
            ).toarray()
        G = nx.from_numpy_array(G)
        clusters = nx_comm.louvain_communities(G, resolution=resolution)
        self.Clusters = clusters
        self.metaData["Cluster"] = np.NaN
        for index, cluster in enumerate(clusters):
            samples = self.Samples[list(cluster)]
            self.metaData.loc[self.Samples[self.Samples.isin(samples)],
                              "Cluster"] = list(string.ascii_uppercase)[index]
        return self
    def umap(self,nNeighbors=15,minDist=0.01):
        '''
        Additional function that executes UMAP projection for visual inspection.
        Does not require PCA or louvain to be executed.

        Parameters:
            - nNeighbors: int, see umap-learn n_neighbors
            - minDist: int, see umap-learn min_dist

        Raises:
            - RuntimeError: Check, if scale was executed beforehand

        Returns:
            - umap-learn object
        '''
        if self.Scaled is None:
            raise RuntimeError("Scaled data not found. Did you forget to run .scale()?")
        reducer = UMAP(n_neighbors=nNeighbors,min_dist=minDist)
        self.Embedding = reducer.fit(self.Scaled[:,self.DepthFeatures])
        return self
    def extractData(self,PCA=True,VarPCA=True,UMAP=True):
        '''
        Extracts additionally generated data as data frame for saving, visualizations, etc.

        Parameters:
            - PCA: True/False, define, if PCA data shall be extracted
            - VarPCA: True/False, define, if explained PCA variance, etc. shall be extracted
            - UMAP: True/False, define, if UMAP data shall be extracted
        '''
        self.Export = {}
        if PCA:
            self.Export["PCA"] = pd.DataFrame(
                data = self.PCA.fit_transform(self.PCA_input),
                index = self.Samples,
                columns = ("PC" + pd.Series(np.arange(self.PCA.n_components_)+1).astype(str))
                )
        if VarPCA:
            self.Export["VarPCA"] = pd.DataFrame(
                data = {
                    "PC":("PC" + pd.Series(np.arange(self.PCA.n_components_)+1).astype(str)),
                    "VarExplained":self.PCA.explained_variance_,
                    "VarExplainedPct":self.PCA.explained_variance_ratio_*100,
                    "CumVarExplained":(self.PCA.explained_variance_ratio_.cumsum()*100)
                    }
                )
        if UMAP:
            self.Export["UMAP"] = pd.DataFrame(
                data = self.Embedding.embedding_,
                index = self.Samples,
                columns = ("UMAP" + pd.Series(np.arange(self.Embedding.n_components)+1).astype(str)))