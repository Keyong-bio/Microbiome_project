import statsmodels.api as sm

# Logistic regression model without interaction
def multiple_glm_model (data, dependent_var, 
                        covariate_sets, 
                        family = sm.families.Binomial()):
    models = {}
    # covariate_sets (is a dictionary)

    for model_name, covariates in covariate_sets.items():
        
        formula = f"{dependent_var} ~ {'+'.join(covariates)}"
        
        print(f"Fitting {model_name} with formula: {formula}")

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")

        # results
        result = model.fit()

        # store the results
        # models[model_name] = result.summary()
        models[model_name] = result
        
    return models

# Make another function to include interactions in the model
def multiple_glm_model_2 (data, dependent_var, covariate_sets, 
                          vars_of_interest = None, 
                          include_interactions = None,
                          family = sm.families.Binomial()):
    models = {}

    for model_name, covariates in covariate_sets.items():
        # start with base covariates
        formula_term = covariates.copy()

        # add only specific interaction
        if vars_of_interest and include_interactions:
            for interaction in include_interactions:
                if interaction not in formula_term:
                    formula_term.append(interaction)

        formula = f"{dependent_var} ~ {'+'.join(formula_term)}"
        print(f'Fitting {model_name} with formula : {formula}')

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")
        result = model.fit()
        models[model_name] = result
    return models

# Consider the non-linear associations between exposure and outcome
# By adding Splines into the function
from patsy import dmatrix

def multiple_glm_model_nonlinear (data, dependent_var, 
                        covariate_sets, 
                        nonlinear_vars = None, 
                        family = sm.families.Binomial()):
    models = {}
    # covariate_sets (is a dictionary)

    for model_name, covariates in covariate_sets.items():

        # linear terms
        linear_vars = '+'.join(covariates)

        # Nonlinear term
        nonlinear_terms = ""
        if nonlinear_vars:
            nonlinear_part = [f"cc({var}, df = {df})" for var, df in nonlinear_vars.items()]
            nonlinear_terms = "+" + "+".join(nonlinear_part)
        
        # formula = f"{dependent_var} ~ {'+'.join(covariates)}"
        # formula = f'{dependent_var} ~ cc({exposure}, df = 4) + {' + '.join(covariates)}'
        formula = f"{dependent_var} ~ {linear_vars}{nonlinear_terms}"
        
        print(f"Fitting {model_name} with formula: {formula}")

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")

        # results
        result = model.fit()

        # store the results
        # models[model_name] = result.summary()
        models[model_name] = result
        
    return models

def logist_model_df(data, dependent_var):
    dfs = []
    for covariates in [covariate_sets_test_AHEI, 
                       covariate_sets_test_rDII, 
                       covariate_sets_test_AMED, 
                       covariate_sets_test_rEDIH,
                  covariate_sets_test_hPDI]:
        
        models_test = multiple_glm_model(data = data, 
                                         dependent_var = dependent_var, 
                                         covariate_sets = covariates)
        
        Exposure = covariates.get("model1", [])[0]
        
        # Also combine the other Dataframes: model1, model2, model3...
        for model in ["model1", "model2", "model3", "model4"]:
            coef = models_test[model].params[Exposure]
            pvalue = models_test[model].pvalues[Exposure]
            conf_int = models_test[model].conf_int(alpha = 0.05).loc[Exposure]
            lower_ci, upper_ci = conf_int
            
           # Create a new Pandas Dataframe
            result_all = pd.DataFrame({
                                "Model": [model],
                                "Exposure" : [Exposure],
                                "Outcome" : [dependent_var],
                                "Coef": [coef],
                                "pval" : [pvalue],
                                "lower_ci" : [lower_ci],
                                "upper_ci" : [upper_ci],
                                "OR": [np.exp(coef)],
                                "lower_OR_ci" : [np.exp(lower_ci)],
                                "upper_OR_ci" : [np.exp(upper_ci)]
                            })
            # dfs.append(result_all)
            dfs = dfs + [result_all]
            
    return dfs

# ## Back up code
# temp1=  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_AHEI_model2,
#                    vars_of_interest = ["AHEI_2010_score_eadj_scaled"],
#                    include_interactions = ["AHEI_2010_score_eadj_scaled:sex_x"])
# display(temp1["model2"].params)

# temp2=  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_rDII_model2,
#                    vars_of_interest = ["rDII_score_eadj_scaled"],
#                    include_interactions = ["rDII_score_eadj_scaled:sex_x"])

# display(temp2["model2"].params)

# temp3 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_AMED_model2,
#                    vars_of_interest = ["AMED_score_eadj_scaled"],
#                    include_interactions = ["AMED_score_eadj_scaled:sex_x"])
# display(temp3["model2"].params)

# temp4 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_rEDIH_model2,
#                    vars_of_interest = ["rEDIH_score_all_eadj_scaled"],
#                    include_interactions = ["rEDIH_score_all_eadj_scaled:sex_x"])
# display(temp4["model2"].params)

# temp5 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_hPDI_model2,
#                    vars_of_interest = ["hPDI_score_eadj_scaled"],
#                    include_interactions = ["hPDI_score_eadj_scaled:sex_x"])
# display(temp5["model2"].params)

# results_df1_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp1["model2"].params,
#         "Con_interval_low": temp1["model2"].conf_int()[0],
#         "Con_interval_upper": temp1["model2"].conf_int()[1],
#         "p_value": temp1["model2"].pvalues
#     }
# )

# results_df2_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp2["model2"].params,
#         "Con_interval_low": temp2["model2"].conf_int()[0],
#         "Con_interval_upper": temp2["model2"].conf_int()[1],
#         "p_value": temp2["model2"].pvalues
#     }
# )

# results_df3_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp3["model2"].params,
#         "Con_interval_low": temp3["model2"].conf_int()[0],
#         "Con_interval_upper": temp3["model2"].conf_int()[1],
#         "p_value": temp3["model2"].pvalues
#     }
# )

# results_df4_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp4["model2"].params,
#         "Con_interval_low": temp4["model2"].conf_int()[0],
#         "Con_interval_upper": temp4["model2"].conf_int()[1],
#         "p_value": temp4["model2"].pvalues
#     }
# )

# results_df5_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp5["model2"].params,
#         "Con_interval_low": temp5["model2"].conf_int()[0],
#         "Con_interval_upper": temp5["model2"].conf_int()[1],
#         "p_value": temp5["model2"].pvalues
#     }
# )

# combined_results_NAFLD = pd.concat([results_df1_NAFLD, results_df2_NAFLD, results_df3_NAFLD, results_df4_NAFLD, results_df5_NAFLD], ignore_index=False)
# combined_results_NAFLD

# # Check the interaction p value 
# combined_results_NAFLD[combined_results_NAFLD.index.str.contains(f"scaled:sex_x")]

# # Backup code
# temp4_1 = multiple_glm_model(data = merge_df_NAFLD,
#                             dependent_var="NAFLD_baseline_diagnosis",
#                             covariate_sets= covariate_sets_test_AHEI_model2)

# temp4_2 =  multiple_glm_model_nonlinear(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                    covariate_sets = covariate_sets_test_AHEI_model2,
#                    nonlinear_vars = {"AHEI_2010_score_eadj_scaled" : 4}            
#                    )

# print(temp4_1["model2"].params)
# print(temp4_2["model2"].pvalues)

# # Compare the two models
# from scipy.stats import chi2

# # Likelihood ratio test
# lr_statistic = 2 * (temp3_1["model2"].llf - temp3_2["model2"].llf)
# lr_statistic
# p_value = 1 - chi2.cdf(lr_statistic, df = 3)
# p_value