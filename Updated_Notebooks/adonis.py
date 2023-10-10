import qiime2 as q2
import pandas as pd
import skbio
import subprocess
import os
import tempfile
from patsy import ModelDesc
from qiime2.plugins import metadata

def _run_command(cmd, verbose=True):
    if verbose:
        print("Running external command line application. This may print "
              "messages to stdout and/or stderr.")
        print("The command being run is below. This command cannot "
              "be manually re-run as it will depend on temporary files that "
              "no longer exist.")
        print("\nCommand:", end=' ')
        print(" ".join(cmd), end='\n\n')
    subprocess.run(cmd, check=True)
    
def adonis(dm, metadata, formula, output_dir, permutations=999, n_jobs=1):
     # Validate sample metadata is superset et cetera
    metadata_ids = set(metadata.ids)
    distance_matrix = dm.view(skbio.DistanceMatrix)
    dm_ids = distance_matrix.ids
    
    # filter metadata to common ids 
    metadata = metadata.filter_ids(set(dm_ids) & metadata_ids)
    
    # filter ids. ids must be in same order as dm
    filtered_md = metadata.to_dataframe() 
    for t in ModelDesc.from_formula(formula).rhs_termlist:
        for i in t.factors: 
            filtered_md = filtered_md.loc[filtered_md[i.name()].notna()]
    
    distance_matrix = distance_matrix.filter(filtered_md.index)
    filtered_md.rename_axis(index='sample_name', inplace=True)
    metadata = q2.Metadata(filtered_md)

    # Run adonis
    results_fp = os.path.join(output_dir, 'adonis.tsv')
    with tempfile.TemporaryDirectory() as temp_dir_name:
        dm_fp = os.path.join(temp_dir_name, 'dm.tsv')
        distance_matrix.write(dm_fp)
        md_fp = os.path.join(temp_dir_name, 'md.tsv')
        metadata.save(md_fp)
        cmd = ['Rscript', 'adonis.R', dm_fp, md_fp, formula, str(permutations),
               str(n_jobs), results_fp]
        _run_command(cmd, verbose=False)
    
    return pd.read_csv(results_fp, sep='\t')

def adonis_and_reformat(distance_matrix, metadata, formula, out_dir='adonis_out'): 
    
    # run adonis 
    out = adonis(distance_matrix, metadata, formula, out_dir) 
    
    variables = out.index.difference(['Residual', 'Total'])
    
    col_names = ['F_stat', 'partial_R2', 'pval', 'formula', 'variable']
    reformat = pd.DataFrame(columns=col_names, index = range(len(variables)))
    for i in range(len(variables)): 
        reformat['F_stat'][i] = out['F'][variables[i]]
        reformat['partial_R2'][i] = out['R2'][variables[i]]
        reformat['pval'][i] = out['Pr(>F)'][variables[i]]
        reformat['variable'][i] = variables[i]
        reformat['formula'][i] = formula 
    
    return reformat