# Untested LLM generated regression code for estimating the causal effect of price on participation rates using a Regression Discontinuity Design (RDD) framework.
# placed for demonstration of RDD modeling

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. GENERATE SYNTHETIC DISTRICT DATA (For Demonstration)
np.random.seed(42)
n_schools = 80

# Generate random ISPs concentrated around the 62.5% margin
isp = np.random.beta(a=5, b=4, size=n_schools) * 0.4 + 0.45 
enrollment = np.random.randint(300, 1200, size=n_schools)
school_type = np.random.choice(['Elementary', 'Middle', 'High'], size=n_schools, p=[0.5, 0.25, 0.25])

df = pd.DataFrame({'isp': isp, 'enrollment': enrollment, 'school_type': school_type})

# Apply the strict district rule: Kicked off CEP if ISP < 62.5%
df['treated'] = (df['isp'] < 0.625).astype(int)

# Center the running variable at the cutoff
df['isp_centered'] = df['isp'] - 0.625

# Simulate Participation Rate (Y) with a sharp drop at the cutoff and school-type friction
def simulate_participation(row):
    # Baseline participation around 75%
    base = 0.75 
    # Structural drop caused by losing CEP (The true causal effect we want to recover)
    price_shock = -0.18 * row['treated'] 
    # Poverty trend effect (higher ISP schools naturally participate slightly more)
    poverty_trend = 0.15 * row['isp_centered'] 
    # Structural grade modifiers
    grade_modifier = 0.0 if row['school_type'] == 'Elementary' else (-0.08 if row['school_type'] == 'Middle' else -0.15)
    
    # Random noise/unobserved variance
    noise = np.random.normal(0, 0.03)
    
    return clamp(base + price_shock + poverty_trend + grade_modifier + noise, 0, 1)

def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)

df['adp_rate'] = df.apply(simulate_participation, axis=1)

# 2. EXECUTE THE RDD REGRESSION MODEL
# We include the interaction term (treated * isp_centered) and control for school type
model_formula = "adp_rate ~ treated * isp_centered + C(school_type, Treatment('Elementary'))"
rdd_model = smf.ols(formula=model_formula, data=df).fit()

# Print the econometric results
print(rdd_model.summary())

# 3. EXTRACT COEFFICIENTS AND CALCULATE ELASTICITY
beta_0 = rdd_model.params['Intercept']
beta_1 = rdd_model.params['treated']  # This is your causal drop

print("\n" + "="*50)
print(class_colors.BOLD + "RDD ANALYSIS OUTPUT:" + class_colors.ENDC)
print(f"Estimated Baseline Participation at Cutoff (CEP/Free): {beta_0:.2%}")
print(f"Causal Drop in Participation due to Price Implementation: {beta_1:.2%}")

# Calculate Arc Elasticity at the Cutoff
# Assume average meal price shifts from $0.00 to a weighted average of $2.50
p_pre = 0.00
p_post = 2.50
p_avg = (p_pre + p_post) / 2

y_pre = beta_0
y_post = beta_0 + beta_1
y_avg = (y_pre + y_post) / 2

pct_delta_y = beta_1 / y_avg
pct_delta_p = (p_post - p_pre) / p_avg

arc_elasticity = pct_delta_y / pct_delta_p
print(f"Calculated Price Elasticity of Demand (ε) near Cutoff: {arc_elasticity:.3f}")
print("="*50)
