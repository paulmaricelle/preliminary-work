import pandas as pd
import scanpy as sc

def get_proteomics(csv_path, only_day_0 = True):
    """returns 2 dataframes, the first one being raw proteomics and the second one being Patient Id, Sex and Age for each sample"""

    df = pd.read_csv(csv_path)
    if only_day_0:
        df_baseline = df[df['sample.visitName'].str.contains('Day 0', na=False)]
    else:
        df_baseline = df

    cols_to_keep = ['sample.sampleKitGuid', 'olink.assay', 'olink.NPX_norm', 'subject.biologicalSex']
    proteomics = df_baseline[cols_to_keep]
    proteomics_pivoted = pd.pivot_table(proteomics, values = "olink.NPX_norm", index = "sample.sampleKitGuid", columns="olink.assay")

    # Extracting useful meta data
    X_meta = df_baseline.drop_duplicates('sample.sampleKitGuid').set_index('sample.sampleKitGuid')
    # We use subjectGuid here so that we can group samples of the same patient together within either the test or train split
    # It has to be removed before training
    X_meta = X_meta[['subject.subjectGuid', 'subject.biologicalSex', 'sample.subjectAgeAtDraw']]

    return proteomics_pivoted, X_meta


def get_genes(path_h5ad, X_df, n_top_genes=None):
    """ Loads Y with log-normalization, get rids of duplicates and aligns it on X """
    adata = sc.read_h5ad(path_h5ad)

    mask_unique = ~adata.obs['sample.sampleKitGuid'].duplicated(keep='first')
    adata = adata[mask_unique].copy()
    
    adata.obs.index = adata.obs['sample.sampleKitGuid']

    common_ids = [guid for guid in X_df.index if guid in adata.obs_names]
    
    if len(common_ids) < len(X_df):
        print(f"There are : {len(X_df) - len(common_ids)} samples in X without corresponding value in Y")
    
    adata_final = adata[common_ids, :].copy()

    # Normalizing the data for each patient
    adata_final.X = adata_final.X.astype('float32')
    sc.pp.normalize_total(adata_final, target_sum=1e4)
    sc.pp.log1p(adata_final)

    # New implementation
    if n_top_genes is not None:
        import numpy as np
        X_dense = adata_final.X.toarray() if hasattr(adata_final.X, 'toarray') else adata_final.X
        gene_var = np.var(X_dense, axis=0)
        top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
        adata_final = adata_final[:, top_idx].copy()
    
    Y_matrix = adata_final.X

    if hasattr(Y_matrix, 'toarray'):
        Y_matrix = Y_matrix.toarray()
        
    return Y_matrix, adata_final.var_names, common_ids

