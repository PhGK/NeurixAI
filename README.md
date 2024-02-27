# Introduction
This markdown file summarizes the datasets used from [Depmap portal](https://depmap.org/portal/)

# Files used
1. **OmicsSomaticMutationsMatrixHotspot.csv**: Genotyped matrix determining for each cell line whether each gene has at least one hot spot mutation. A variant is considered a hot spot if it's present in one of the following: Hess et al. 2019 paper, OncoKB hotspot, COSMIC mutation significance tier 1. (0 == no mutation; If there is one or more hot spot mutations in the same gene for the same cell line, the allele frequencies are summed, and if the sum is greater than 0.95, a value of 2 is assigned and if not, a value of 1 is assigned.)
2. **OmicsExpressionProteinCodingGenesTPMLogp1.csv**: Gene expression TPM values of the protein coding genes for DepMap cell lines. These values are inferred from RNA-seq data using the RSEM tool and are reported after log2 transformation, using a pseudo-count of 1 (log2(TPM+1)). Additional RNA-seq-based expression measurements are available for download as part of the full DepMap Data Release. More information on the DepMap Omics processing pipeline is available at [DepMap Omics processing pipeline](https://github.com/broadinstitute/depmap_omics).

   - Genes: 19193
   - Cell Lines: 1479
   - Primary Diseases: 76
   - Lineages: 32
   - Source: Broad Institute

3. **secondary-screen-replicate-treatment-info.csv**: List of all drugs in secondary screening with metadata like smiles codes and more. 
4. **Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv**: All drug names, targets and pathways.

### PRISM 
- [Nature article on anticancer potential of non-oncology drugs by systematic viability profiling](https://www.nature.com/articles/s43018-019-0018-6)

![Bildschirmfoto 2024-02-26 um 18 35 21](https://github.com/NiklasKiermeyer/DruxAI/assets/44393665/01f9278d-7701-46f9-a969-be1ede1bab5a)
