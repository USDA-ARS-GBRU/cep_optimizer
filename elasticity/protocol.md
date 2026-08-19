# Protocol: RDD Estimation of Student Meal Demand Elasticity

## Step 1: Define the Variables

* **Running Variable (Forcing Variable):** Individual School ISP ($X_{s}$).
* **Policy Cutoff ($c$):** 62.5% ($0.625$). Schools with $X_{s} < 0.625$ lose CEP status and charge cash; schools with $X_{s} \ge 0.625$ retain 100% free meals.
* **Treatment Variable ($D_{s}$):** A binary indicator where $D_{s} = 1$ if $X_{s} < 0.625$ (Exited CEP/Charged Price) and $D_{s} = 0$ if $X_{s} \ge 0.625$ (Remained CEP/Free).
* **Centered Running Variable ($\tilde{X}_{s}$):** $X_{s} - 0.625$. Centering places the policy threshold exactly at zero, allowing direct interpretation of the intercept shift.
* **Outcome Variable ($Y_{s}$):** Average Daily Participation (ADP) Rate, calculated as $\frac{\text{Average Daily Meals Served}}{\text{Total School Enrollment}}$.


## Step 2: Resolve the Co-mingled Demand Asymmetry (The Three-Group Mixture Model)

During CEP years, register data only tracks *Total Meals Served* $Y_{total, s}$, hiding the individual behavior of the three distinct demographic groups ($g$):
1.  **Free Eligible ($f$):** Directly certified via SNAP/Medicaid or income applications.
2.  **Reduced Eligible ($r$):** Households between 130% and 185% of the federal poverty line.
3.  **Paid Tier ($p$):** Full-price, higher-income students.

To isolate true elasticity, your model must recognize that **Total Demand is a weighted sum of three distinct, group-specific participation rates ($y_g$)**:

$$
Y_{total, s} = \omega_{f,s} \cdot y_{f,s} + \omega_{r,s} \cdot y_{r,s} + \omega_{p,s} \cdot y_{p,s}
$$

Where $\omega_{g,s}$ represents the known enrollment percentage of that demographic group at school $s$ (harvested from your post-exit data and historical direct certification records).

### The Group Behavioral Constraints:

*   **The Free Group ($y_f$):** Out-of-pocket price is \$0.00 both before and after the policy shift. Any drop in their participation is entirely driven by administrative/paperwork friction ($\delta_f$), not price elasticity.
*   **The Reduced Group ($y_r$):** Out-of-pocket price shifts from $0.00 to $0.40. They have low-to-moderate price sensitivity.
*   **The Paid Group ($y_p$):** Out-of-pocket price spikes from \$0.00 to \$2.50/\$2.75. They possess the highest price elasticity ($\epsilon_p$) and drive the vast majority of the total volume crash.

To impute the missing historical baseline, the model holds the post-exit structural relationships constant to back-cast the group-specific participation rates ($y_{f,pre}, y_{r,pre}, y_{p,pre}$) that mathematically sum to your known historical $Y_{total, pre}$.


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
