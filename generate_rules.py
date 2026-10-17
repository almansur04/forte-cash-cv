import pandas as pd

# Mapping of Visual Cues to Regulatory Justifications
data = [
    {
        "violation_type": "unattended_cash_drawer",
        "visual_cue": "Cashier leaves the frame while the cash drawer/safe remains visibly open.",
        "presumed_violation": "Failure to secure valuables/lock the cash repository when leaving the workplace.",
        "regulatory_act": "NBRK Resolution No. 231 (29.11.2019)",
        "clause_number": "TBD - User to verify",
        "adilet_link": "https://adilet.zan.kz/rus/docs/V1900019680"
    },
    {
        "violation_type": "unauthorized_person_in_zone",
        "visual_cue": "Two or more people detected simultaneously inside the cashier's secure workspace (behind the counter).",
        "presumed_violation": "Presence of unauthorized persons in the cash desk zone.",
        "regulatory_act": "NBRK Resolution No. 231 (29.11.2019)",
        "clause_number": "TBD - User to verify",
        "adilet_link": "https://adilet.zan.kz/rus/docs/V1900019680"
    },
    {
        "violation_type": "exposed_cash_on_counter",
        "visual_cue": "Cash bundles left on the counter/table unattended by the cashier.",
        "presumed_violation": "Improper storage of cash during operations.",
        "regulatory_act": "NBRK Resolution No. 231 (29.11.2019)",
        "clause_number": "TBD - User to verify",
        "adilet_link": "https://adilet.zan.kz/rus/docs/V1900019680"
    }
]

df_rules = pd.DataFrame(data)
df_rules.to_csv("violation_rules.csv", index=False)
print("Created violation_rules.csv")
print("ACTION REQUIRED: Open the Adilet link, find the specific clause numbers, and update the CSV.")