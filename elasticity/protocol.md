# Protocol: RDD Estimation of Student Meal Demand Elasticity

## Step 1: Define the Variables

* **Running Variable (Forcing Variable):** Individual School ISP ($X_{s}$).
* **Policy Cutoff ($c$):** 62.5% ($0.625$). Schools with $X_{s} < 0.625$ lose CEP status and charge cash; schools with $X_{s} \ge 0.625$ retain 100% free meals.
* **Treatment Variable ($D_{s}$):** A binary indicator where $D_{s} = 1$ if $X_{s} < 0.625$ (Exited CEP/Charged Price) and $D_{s} = 0$ if $X_{s} \ge 0.625$ (Remained CEP/Free).
* **Centered Running Variable ($\tilde{X}_{s}$):** $X_{s} - 0.625$. Centering places the policy threshold exactly at zero, allowing direct interpretation of the intercept shift.
* **Outcome Variable ($Y_{s}$):** Average Daily Participation (ADP) Rate, calculated as $\frac{\text{Average Daily Meals Served}}{\text{Total School Enrollment}}$.

## Step 2: Handle the Historical Data Asymmetry

Because individual meal categories (Free/Reduced/Paid) are unobserved during CEP years, you must isolate the total school-level demand shock at the cutoff, and then scale it by the school’s underlying demographic profile.

1. Match each school's post-CEP data with its historical pre-exit data.
2. Use **Synthetic Cohort Back-Casting** to estimate the baseline paid-eligible demographic pool for the CEP years.

## Step 3: Specify the RDD Econometric Model

To allow for the possibility that the relationship between ISP and participation differs on either side of the 62.5% threshold, use a Linear RDD with Interaction Terms:

$$
Y_{s} = \beta_{0} + \beta_{1}D_{s} + \beta_{2}\tilde{X}_{s} + \beta_{3}(D_{s} \times \tilde{X}_{s}) + \gamma \mathbf{Z}_{s} + \epsilon_{s}
$$

Where:

* **$\beta_{0}$:** The baseline participation rate for a school sitting exactly at the 62.5% threshold that remained on CEP.
* **$\beta_{1}$:** The Causal Effect of the Price Shock. This represents the vertical drop in participation rate right at the threshold boundary.
* **$\beta_{2}$:** The slope of the participation trend for schools that stayed on CEP.
* **$\beta_{3}$:** The change in the slope for schools that were kicked off CEP.
* **$\mathbf{Z}_{s}$:** A vector of structural control variables (School Type Dummies: Middle, High).

## Step 4: Calculate the Arc Price Elasticity ($\epsilon$)

Once $\beta_{1}$ is estimated, translate this absolute drop in participation into an elasticity coefficient:

$$
\epsilon = \frac{\% \Delta \text{ Participation}}{\% \Delta \text{ Price}} = \frac{\beta_{1} / \beta_{0}}{(P_{\text{post}} - P_{\text{pre}}) / P_{\text{avg}}}
$$

Since $P_{\text{pre}} = \$0.00$, the percent change in price is technically infinite if calculated relative to the baseline. Therefore, you must use Arc Elasticity (using the average price and average participation at the cutoff boundary) to keep the denominator stable.
