import pandas as pd

drug_interactions = pd.read_csv('drug_interaction_dataset_900.csv')

def check_interaction(drug1, drug2):
    # Check if there's an interaction between drug1 and drug2
    interaction = drug_interactions[
        ((drug_interactions['drug1'] == drug1) & (drug_interactions['drug2'] == drug2)) |
        ((drug_interactions['drug1'] == drug2) & (drug_interactions['drug2'] == drug1))
    ]
    if not interaction.empty:
        return interaction.iloc[0]['interaction'], interaction.iloc[0]['severity']
    return None, None