import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
import networkx as nx
import networkx.algorithms.community as nx_comm
class SampleCommunitiesPCA:
    '''
    A class for performing PCA-based clustering of bulk RNA-seq samples.
    The general flow is based on scRNA-seq analysis, like in SEURAT for R
    '''
    def __init__(self,logCounts,normCounts,metaData,groupVar=None,group=None):
        '''
        Initialize the SampleCommunitiesPCA class.

        Parameters:
            - logCounts: DataFrame, log-transformed gene expression data.
            - normCounts: DataFrame, normalized gene expression data.
            - metaData: DataFrame, metadata for samples.
            - groupVar: str, variable for grouping samples (optional).
            - group: str, specific group of samples to analyze (required, if groupVar != None).

        Raises:
            - ValueError: metaData, normCount and logCount Sample IDs (index for metaData, columns for rest) have to match,
                otherwise the results will not make sense for visualization, etc; a common problem for beginners.
        '''
        if not logCounts.columns.equals(metaData.index) or not normCounts.columns.equals(metaData.index):
            raise ValueError("CountMatrix and MetaData do not match. They should have the same sample IDs.")
        self.logCounts = logCounts
        self.normCounts = normCounts
        self.group = group
        self.groupVar = groupVar
        if groupVar != None:
            if group == None:
                raise RuntimeError("No group is defined. Please set groupVar = None, or define group")
            self.Samples = metaData.loc[metaData[groupVar] == group,:].index.array
        else:self.Samples = metaData.index.array
        self.metaData = metaData.loc[self.Samples,:]
    def scale(self,with_std=True):
        '''
        Required scaling of the data using sklearns StandardScaler

        Parameters:
            - with_sd: True/False, gives user choice to center or scale the data

        Note: scaling is highly encouraged as it usually improves results.

        '''
        scaler = StandardScaler(with_mean=True,with_std=with_std)
        self.Scaled = scaler.fit_transform(self.logCounts.T.loc[self.Samples,:])
        return self
    def depthFilter(self,thrsh=50):
        '''
        A required function that selects features (genes) based on normalized read depth.

        Filtering for depth of RNA-seq data is very important, expression of 0 is most common and
            low count genes tend to have extremely high variance and worsen analysis.

        Parameters:
        - thrsh: int, threshold for depth filtering, default 50 is a good heuristic

        Returns:
            - DepthFeatures: array of int, Indices of genes with mean expression > thrsh
        '''
        tmp=self.normCounts.loc[:,self.Samples].mean(axis=1)>thrsh
        self.DepthFeatures = self.normCounts.reset_index().index.array[tmp]
        self.DepthGenes = self.normCounts.index.array[tmp]
        return self
    def pca(self,nFeatures):
        '''
        Parameters:
            - nFeatures : int, The number of features to be used for PCA.
            Selected are the top nFeatures with highest variance after filtering for depth.
            Variance is based on logCounts as normCounts are heteroskedastic

        Raises:
            - RuntimeError: Check, if depthFilter and scale were executed, as they are required.

        Returns:
            - Features: array of int, Indices of genes to use as input for PCA.
            - Genes: array of str, gene IDs of selected Features.
            - PCA_input: array, input for PCA.
            - PCA: sklearn object
        '''
        if self.Scaled is None:
            raise RuntimeError("Scaled data not found. Did you forget to run .scale() ?")
        if self.DepthFeatures is None:
            raise RuntimeError("DepthFeatures not found. Did you forget to run .depthFilter() ?")
        geneVariances = np.var(self.logCounts.loc[self.DepthGenes,self.Samples].values, axis = 1)
        self.Features = np.argsort(geneVariances)[-nFeatures:]
        self.Genes = self.logCounts.iloc[self.Features,:].index.tolist()
        self.PCA_input = self.Scaled[:,self.Features]
        pca = PCA(n_components=min(len(self.Samples),len(self.Features)))
        self.PCA = pca.fit(self.PCA_input)
        return self
    def louvain(self,minPCs=10,minPctVar=90,nNeighbors=10,metric="euclidean",resolution=1):
        '''
        Parameters:
            - minPCs: int, The minimum number of PCs to be used for graph generation
            - minPctVar: float, the minimum variance to be explained by the input
            - nNeighbors: int, the number of neighbors for kNN graph
            - resolution: float, resolution used by the louvain algorithm

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
        print(f"Using {pcs} PCs, explaining {cumVar[pcs]} % of the total variance")
        self.PCs = pcs
        self.PCA_output = self.PCA.fit_transform(self.PCA_input)[:,[*range(pcs)]]
        G = kneighbors_graph(
            self.PCA_output,
            n_neighbors = nNeighbors,
            metric=metric
            ).toarray()
        G = nx.from_numpy_array(G)
        clusters = nx_comm.louvain_communities(G, resolution=resolution)
        self.Clusters = clusters
        self.metaData["Cluster"] = np.NaN
        for index, cluster in enumerate(clusters):
            samples = self.Samples[list(cluster)]
            self.metaData.loc[self.Samples[self.Samples.isin(samples)],
                              "Cluster"] = f"C{index}"
        return self
    def umap(self,nNeighbors=15,minDist=0.01,metric="euclidean"):
        '''
        Additional function that executes UMAP projection for visual inspection.
        Does require PCA but not louvain to be executed.

        Parameters:
            - nNeighbors: int, see umap-learn n_neighbors
            - minDist: int, see umap-learn min_dist

        Raises:
            - RuntimeError: Check, if scale and depthFilter were executed beforehand

        Returns:
            - umap-learn object
        '''
        if self.Scaled is None:
            raise RuntimeError("Scaled data not found. Did you forget to run .scale()?")
        if self.DepthFeatures is None:
            raise RuntimeError("DepthFeatures not found. Did you forget to run .depthFilter()?")
        reducer = UMAP(n_neighbors=nNeighbors,min_dist=minDist,metric=metric)
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