from pkg_resources import resource_filename

import biom
from birdman import SingleFeatureModel
import numpy as np
import pandas as pd

MODEL_PATH = "/projects/u19/Tulsa/birdman/src/stan/negative_binomial_single.stan"
MD = pd.read_table("/projects/u19/Tulsa/data/metadata_anxiety.tsv", sep="\t", index_col=0)

class AnxietyModelSingle(SingleFeatureModel):
    def __init__(
        self,
        table: biom.Table,
        feature_id: str,
        beta_prior: float = 5.0,
        inv_disp_sd: float = 0.5,
        num_iter: int = 500,
        num_warmup: int = 500,
        **kwargs
    ):
        super().__init__(
            table=table,
            feature_id=feature_id,
            model_path=MODEL_PATH,
            num_iter=num_iter,
            num_warmup=num_warmup,
            **kwargs
        )


        D = table.shape[0]
        A = np.log(1 / D) 
	# build formula
        self.create_regression(formula="PROMIS_AnxietyTscore+Gender+Age", metadata=MD)

        param_dict = {
            "depth": np.log(table.sum(axis="sample")),
            "B_p": beta_prior,
            "inv_disp_sd": inv_disp_sd,
	    "A": A
        }
        self.add_parameters(param_dict)

        self.specify_model(
            params=["beta_var", "inv_disp"],
            dims={
                "beta_var": ["covariate"],
                "log_lhood": ["tbl_sample"],
                "y_predict": ["tbl_sample"]
            },
            coords={
                "covariate": self.colnames,
                "tbl_sample": self.sample_names,
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"

        )
