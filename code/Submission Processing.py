"""Run this file when ready to submit file"""


# Shaping dataset for submission.
def melt_prods_id(df):
    rearranged_df = df[PROD_CODES].copy()
    rearranged_df.insert(loc=0, column='ID', value=test_id)
    melted_df = pd.melt(rearranged_df,id_vars='ID',value_vars=PROD_CODES,
                        var_name='PCODE', value_name='Label', ignore_index=True)
    return melted_df

# This function will identify which products were purchased already. To replace the predicted_probabilities with ones.
def ones_for_purchased_prods(predictions, vanilla):
    melted_pred = melt_prods_id(predictions) # Previously defined function.
    melted_vanilla = melt_prods_id(vanilla)
    mask_index = melted_vanilla[(melted_vanilla['Label'] == 1)].index
    melted_pred.loc[mask_index,'Label'] = 1
    ones_replaced = melted_pred.sort_values('ID')
    return ones_replaced

def format_for_submission(df_to_format):
    replaced_ones = df_to_format
    replaced_ones['ID X PCODE'] = replaced_ones['ID'] + ' X ' + replaced_ones['PCODE']
    submission = replaced_ones[['ID X PCODE', 'Label']].copy()
    return submission

replaced_with_ones = ones_for_purchased_prods(pred_df, test)
for_submission = format_for_submission(replaced_with_ones)
for_submission.to_csv('Submission_CB_2.csv', index=False,header=True)
