# NGS-Analysis

## Overview

 Classes and re-usable scripts for NGS data analysis

## Contents

### modules

Contains Python modules for NGS data analysis.

1. CoRegulatoryNetworkAnalysis.py:
    - Class for clustering gene expression from RNA-seq data
    - Produces graph based on pearson correlation
    - Executes community detection (Louvain)

2. KNNetworkClustering.py
    - Class for clustering gene expression from RNA-seq data
    - Produces graph based on k-nearest neighbors
    - Executes community detection (Louvain)

3. SampleCommunitiesPCA.py
    - Clustering of Samples from an RNA-seq experiment
        - Similar idea as for scRNA-seq data analysis in Seurat (https://github.com/satijalab/seurat)
    - Performs PCA
    - Produces graph based on k-nearest neighbors
    - Executes community detection (Louvain)

4. SamplePCA.py
    - Class for PCA analysis of RNA-seq samples

5. SampleUMAP.py
    - Class for UMAP visualization of RNA-seq samples

6. SNPSampleCommunitiesPCA.py
    - Determination of population structures from SNP data
    - Performs PCA
    - Produces graph based on k-nearest neighbors
    - Executes community detection (Louvain)
