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
5. **secondary-screen-dose-response-curve-parameters.csv**: Parameters of dose-response curves fit to replicate-level viability data using a four-parameter log-logistic function.

### PRISM

- [Nature article on anticancer potential of non-oncology drugs by systematic viability profiling](https://www.nature.com/articles/s43018-019-0018-6)

![Bildschirmfoto 2024-02-26 um 18 35 21](https://github.com/NiklasKiermeyer/DruxAI/assets/44393665/01f9278d-7701-46f9-a969-be1ede1bab5a)

# Pharmacogenomic Landscape Interactions for GDSC1/2 Dataset

- [Cell article](https://www.sciencedirect.com/science/article/pii/S0092867416307462)
- Integration of heterogeneous molecular data from 11,289 tumors and 1,001 cell lines
- Measurement of the response of 1,001 cancer cell lines to 265 anti-cancer drugs
- Discovery of numerous oncogenic aberrations that sensitize to anti-cancer drugs
- Provision of a valuable resource to identify therapeutic options for cancer sub-populations

<img width="576" alt="Bildschirmfoto 2024-03-05 um 08 16 37" src="https://github.com/NiklasKiermeyer/DruxAI/assets/44393665/4287677e-14ec-4557-90b4-670390d9e1f3">
<img width="854" alt="Bildschirmfoto 2024-03-05 um 08 56 47" src="https://github.com/NiklasKiermeyer/DruxAI/assets/44393665/e71a14c2-ed31-4804-9daa-a957a5025734">
<img width="743" alt="Bildschirmfoto 2024-03-05 um 09 13 44" src="https://github.com/NiklasKiermeyer/DruxAI/assets/44393665/e9d50d4a-d478-44a0-b943-8c01ce687586">

### Additional Data Resources

- [Chemistry Library](https://www-library.ch.cam.ac.uk/list-useful-databases)
- [CTR-DB, an omnibus for patient-derived gene expression signatures correlated with cancer drug response](https://pubmed.ncbi.nlm.nih.gov/34570230/)
- [CTR-DB, an omnibus for patient-derived gene expression signatures correlated with cancer drug response](https://academic.oup.com/nar/article/50/D1/D1164/6389514)

### Papers

- [Predicting cellular responses to complexperturbations in high-throughput screens](https://www.embopress.org/doi/epdf/10.15252/msb.202211517)
- [scGen predicts single-cell perturbation responses](https://www.nature.com/articles/s41592-019-0494-8)

### Data profiling

- [TargetDB](https://github.com/sdecesco/targetDB)
