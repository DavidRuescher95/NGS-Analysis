import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import networkx.algorithms.community as nx_comm
from umap.umap_ import UMAP
from scipy.stats import rankdata
class CRN:
    '''
    A class for performing Community detection on RNA-seq data based on a graph adjacency matrix created
    through pearson correlation
    '''
    def __init__(self,logCounts,normCounts,metaData,LRT=None,groupVar=None,group=None):
        '''
        Initialize the CRN class.

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
        else:
            self.logCounts = logCounts
            self.normCounts = normCounts
        self.group = group
        self.groupVar = groupVar
        if groupVar != None:
            if group == None:
                raise RuntimeError("No group is defined. Please set groupVar = None, or define group")
            self.Samples = metaData.loc[metaData[groupVar] == group,:].index.array
            if LRT is not None:
                self.lrt = LRT.loc[LRT[groupVar]==group,:]
        else:
            self.Samples = metaData.index.array
            if LRT is not  None:
                self.lrt = LRT
        self.metaData = metaData.loc[self.Samples,:]
    def scale(self,with_mean=True,with_std=True):        
        scaler = StandardScaler(with_std=with_std)        
        self.Scaled = scaler.fit_transform(self.logCounts.T.loc[self.Samples,:]).T       
        return self
        '''
        Required scaling of the data using sklearns StandardScaler

        Parameters:
            - with_sd: True/False, gives user choice to center or scale the data

        Note: scaling is highly encouraged as it usually improves results.

        '''
    def depthFilter(self,thrsh=50):
        '''
        Required function, if corrNetwork shall be depth based.
        Selects features (genes) based on normalized read depth.

        Filtering for depth of RNA-seq data is very important, expression of 0 is most common and
            low count genes tend to have extremely high variance and worsen analysis.

        Parameters:
        - thrsh: int, threshold for depth filtering, default 50 is a good heuristic

        Returns:
            - DepthFeatures: array of int, Indices of genes with mean expression > thrsh
        '''
        tmp=self.normCounts.loc[:,self.Samples].mean(axis=1)>thrsh
        self.DepthFeatures = self.normCounts.reset_index().index.array[tmp]
        return self
    def pValFilter(self,thrsh=0.001):
        '''
        Required function, if corrNetwork shall be pValue based.
        Selects features (genes) based on adjusted pValues reported by DESeq2 LRT.

        Filtering for depth of RNA-seq data is done by DESeq2. No need here.

        Parameters:
        - thrsh: int, threshold for pValue filter. DESeq2 FDR is very lenient, choose stringent cutoff 

        Returns:
            - pValFeatures: array of int, Indices of genes with pAdj < thrsh
        '''
        tmp=self.lrt.padj < thrsh
        self.pValFeatures = self.normCounts.reset_index().index.array[tmp]
        return self
    def corrNetwork(self,nPerm=1000,filterType=["pValue","depth"],corrThrsh="permutation",quantileCutoff=0.995,
                    runPCA = False, minPCs=10, minPctVar=90):
        if not hasattr(self, "Scaled"):
            raise RuntimeError("Scaled not found. Did you forget to run .scale() ?")
        if filterType == "pValue":
            if not hasattr(self, "pValFeatures"):
                raise RuntimeError("pValFeatures not found. Did you forget to run .pValFilter() ?")
            features = self.pValFeatures
        elif filterType == "depth":
            if not hasattr(self, "DepthFeatures"):
                raise RuntimeError("DepthFeatures not found. Did you forget to run .depthFilter() ?")
            features = self.DepthFeatures
        if runPCA:
            print("Executing PCA")
            pcaInput = self.Scaled[features,:]
            pca = PCA(n_components=min(pcaInput.shape[1],len(self.Samples)))
            pca.fit(pcaInput)
            cumVar = pca.explained_variance_ratio_.cumsum()
            pcs = max(
                len(cumVar[cumVar<(minPctVar/100)])+1,
                minPCs
                 )
            print(f"Using {pcs} PCs, explaining {cumVar[pcs]*100} % of the total variance")
            data = pca.fit_transform(pcaInput)[:,[*range(pcs)]]
        else:
            data = self.Scaled[features,:]
        if corrThrsh=="permutation":
            CorrThrsh = np.zeros(nPerm)
            k=0
            while k < nPerm:
                print(k)
                tmp = data[np.arange(len(data))[:,None], np.random.randn(*data.shape).argsort(axis=1)]
                tmp = np.corrcoef(tmp)
                CorrThrsh[k] = np.quantile(tmp[np.triu_indices(tmp.shape[1],k=1)],quantileCutoff)
                k=k+1
            self.CorrThrsh = np.median(CorrThrsh)
        else: self.CorrThrsh=corrThrsh
        corr = np.corrcoef(data)
        self.CorMatrix = corr
        G = corr-self.CorrThrsh
        G[G>0] = 1
        G[G<=0] = 0
        self.CorrNetwork = G
        self.NetworkFeatures = features
        return self
    def networkGeneration(self, remove = False):
        '''
        Required function to generate the Graph on which community detection will
        be executed.

        Parameters:
        - remove: logical, define if the original network shall be removed to save memory

        Returns:
            - G: networkx Graph
        '''
        if not hasattr(self, "CorrNetwork"):
            raise RuntimeError("CorrNetwork not found. Did you forget to run .corrNetwork() ?")
        print("Generating graph from Network. This might take a while...")
        self.G = nx.from_numpy_array(self.CorrNetwork)
        if remove is True:
            delattr(self, "CorrNetwork")
    def louvain(self,minEdges=50, minClusterSize=50, minClusterDegreeRatio = 0.1, resolution=1):
        '''
        Function that executes community detection on the generated network.

        Parameters:
        - minEdges: int, the minimum degree for a node to be included in the analysis
        - minClusterSize: clusters with fewer nodes will be discarded
        - minClusterDegreeRatio: float [0,1], nodes with fewer connections within the cluster will be removed
        - resolution: float, resolution used by the louvain algorithm; higher will lead to more clusters

        Returns:
            - Clusters: array of int, indicates cluster for each gene in NetworkFeatures
        '''
        if not hasattr(self, "G"):
            raise RuntimeError("Graph not found. Did you forget to run .networkGeneration() ?")
        remove = [node for node,degree in dict(self.G.degree()).items() if degree < minEdges]
        removedGenes = self.logCounts.index.array[self.NetworkFeatures[list(remove)]]
        print(f"Removing {len(remove)} genes: low connectivity")
        self.G.remove_nodes_from(remove)
        print("Executing Louvain algorithm. This might take a while...")
        clusters = nx_comm.louvain_communities(self.G, resolution=resolution)
        clusters.sort(key=len)
        outliers = pd.DataFrame(data={"CommunityDegree":np.nan,
                           "Cluster":-1}, index = removedGenes)
        tmp = []
        for index, cluster in enumerate(clusters): 
            print(index)
            g = self.G.subgraph(list(cluster))
            nGenes = g.number_of_nodes()
            removeCommunity = [node for node,degree in dict(g.degree()).items() if (degree/nGenes) < minClusterDegreeRatio]
            removedGenesCommunity = self.logCounts.index.array[self.NetworkFeatures[list(removeCommunity)]]
            community = pd.DataFrame(data={"CommunityDegree":[val for (node, val) in g.degree()],
                               "Cluster":index}, index = self.logCounts.index.array[self.NetworkFeatures[list(cluster)]])
            if nGenes >= minClusterSize:
                print(f"Removing {len(removeCommunity)} genes: low connectivity within community")
                community.loc[removedGenesCommunity,"Cluster"] = -1
                tmp.append(community)
            else: 
                print(f"removing cluster: number of genes < {minClusterSize}")
                community.Cluster = -1
        tmp.append(outliers)
        self.Clusters = pd.concat(tmp)
        self.Clusters.sort_values(["Cluster"])
        return self
    def umap(self,nNeighbors=15,minDist=0.01,metric="euclidean"):
        '''
        Additional function that executes UMAP projection for visual inspection.

        Parameters:
            - nNeighbors: int, see umap-learn n_neighbors
            - minDist: int, see umap-learn min_dist

        Raises:
            - RuntimeError: Check, if scale and networkGeneration were executed

        Returns:
            - umap-learn object
        '''
        if self.Scaled is None:
            raise RuntimeError("Scaled data not found. Did you forget to run .scale()?")
        if self.NetworkFeatures is None:
            raise RuntimeError("NetworkFeatures not found. Did you forget to run .networkGeneration()?")
        reducer = UMAP(n_neighbors=nNeighbors,min_dist=minDist,metric=metric)
        self.Embedding = reducer.fit(self.Scaled[self.NetworkFeatures,:])
        return self
    def extractData(self,Network=False,UMAP=True,Scaled=True):
        self.Export = {}
        if Network:
            self.Export["Network"] = pd.DataFrame(
                data = self.CorrNetwork,
                index = self.logCounts.index.array[self.NetworkFeatures],
                columns = self.logCounts.index.array[self.NetworkFeatures]
                )
        if UMAP:
            self.Export["UMAP"] = pd.DataFrame(
                data = self.Embedding.embedding_,
                index = self.logCounts.index.array[self.NetworkFeatures],
                columns = ("UMAP" + pd.Series(np.arange(self.Embedding.n_components)+1).astype(str)))
        if Scaled:
            self.Export["Scaled"] = pd.DataFrame(
                data=self.Scaled,index=self.normCounts.index,columns=self.Samples)
