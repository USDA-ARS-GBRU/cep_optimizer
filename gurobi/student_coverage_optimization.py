import pandas as pd
import math
from gurobipy import Model, GRB, quicksum

def read_and_prepare_data(csv_file):
    """
    Reads the CSV file and prepares the school data.
    Computes:
      - enrollment (k)
      - ISP as a fraction (from percentage string)
      - isp_count = round(ISP * enrollment)
    Returns a DataFrame and a list of school indices.
    """
    df = pd.read_csv(csv_file)
    # Clean ISP: remove the "%" symbol and convert to fraction
    def isp_to_fraction(isp_str):
        try:
            return float(isp_str.strip().strip("%")) / 100.0
        except Exception:
            return 0.0
    df["ISP_fraction"] = df["ISP"].apply(isp_to_fraction)
    df["ENROLLMENT"] = pd.to_numeric(df["ENROLLMENT"], errors="coerce")
    # Compute isp_count as rounding of ISP * enrollment
    df["isp_count"] = df.apply(lambda row: int(round(row["ISP_fraction"] * row["ENROLLMENT"])), axis=1)
    return df

def solve_optimization(df):
    """
    Solves the optimization problem using gurobipy:
    - Schools are assigned to groups (max groups = number of schools)
    - Each school must be assigned to exactly one group.
    - For each group g:
         Let E_g = sum(enrollment of schools in group g)
         Let I_g = sum(isp_count of schools in group g)
      Define binary variable y_g which is 1 if group qualifies (i.e. if I_g >= 0.25 * E_g).
      Introduce z_g, a linearized variable representing E_g if group qualifies; otherwise 0.
      The objective is to maximize the sum of z_g over all groups.
    """
    schools = df.index.tolist()
    n = len(schools)
    groups = range(n)  # Maximum groups equal to number of schools

    # Total enrollment sum for big-M computation
    M = df["ENROLLMENT"].sum()

    # Create model
    model = Model("school_grouping")

    # Binary assignment variable: x[s, g] == 1 if school s is assigned to group g
    x = {}
    for s in schools:
        for g in groups:
            x[s, g] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{g}")

    # Binary indicator for each group qualifying (group ISP >= 0.25)
    y = {}
    for g in groups:
        y[g] = model.addVar(vtype=GRB.BINARY, name=f"y_{g}")
        
    # Continuous variable z[g] representing the group enrollment if qualifies, 0 otherwise.
    z = {}
    for g in groups:
        z[g] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{g}")
    
    model.update()

    # Constraint: each school is assigned to exactly one group.
    for s in schools:
        model.addConstr(quicksum(x[s, g] for g in groups) == 1, name=f"school_assignment_{s}")

    # For each group, compute group enrollment and isp_count as expressions.
    # Then add constraints linking qualification and the z variable.
    for g in groups:
        # Group total enrollment E_g and total ISP count I_g.
        E_g = quicksum(df.loc[s, "ENROLLMENT"] * x[s, g] for s in schools)
        I_g = quicksum(df.loc[s, "isp_count"] * x[s, g] for s in schools)
        
        # Constraint to enforce qualification: if y[g]==1 then I_g must be at least 0.25 * E_g.
        # Using big-M: I_g >= 0.25 * E_g - M*(1 - y[g])
        model.addConstr(I_g >= 0.25 * E_g - M * (1 - y[g]), name=f"qualify_{g}")
        
        # Linearization constraints for z[g] = E_g * y[g]:
        model.addConstr(z[g] <= E_g, name=f"z_upper1_{g}")
        model.addConstr(z[g] <= M * y[g], name=f"z_upper2_{g}")
        model.addConstr(z[g] >= E_g - M * (1 - y[g]), name=f"z_lower_{g}")
    
    # Set objective: maximize the sum of enrollment in groups that meet the ISP criteria.
    model.setObjective(quicksum(z[g] for g in groups), GRB.MAXIMIZE)

    model.optimize()

    # Extract solution: assignment for each school.
    assignment = {}
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
        for s in schools:
            for g in groups:
                if x[s, g].X > 0.5:
                    assignment[s] = g
                    break

    solution_objective = model.ObjVal if model.status == GRB.Status.OPTIMAL else None

    return solution_objective, assignment, model

def compute_comparisons(df):
    """
    Computes comparison benchmarks:
    1. When all schools are in one group.
    2. When each school is in its own group.
    
    For a group, the group qualifies if (sum of isp_count) >= 0.25*(sum of enrollment).
    """
    # 1. All schools in one group:
    total_enrollment = df["ENROLLMENT"].sum()
    total_isp_count = df["isp_count"].sum()
    qualifies_one_group = (total_isp_count >= 0.25 * total_enrollment)
    one_group_value = total_enrollment if qualifies_one_group else 0

    # 2. Each school in its own group:
    # Each school qualifies individually if its isp_count >= 0.25*enrollment.
    individual_value = 0
    for idx, row in df.iterrows():
        if row["isp_count"] >= 0.25 * row["ENROLLMENT"]:
            individual_value += row["ENROLLMENT"]
    return one_group_value, individual_value

def main():
    # Update the CSV file name accordingly.
    csv_file = "example.csv"
    df = read_and_prepare_data(csv_file)
    
    # Solve the optimization model.
    opt_value, assignment, model = solve_optimization(df)
    print("Optimization Results:")
    print(f"Optimal Objective (Total enrollment for qualified groups): {opt_value}")
    
    # Display school assignments.
    print("\nSchool Group Assignments:")
    # Group by group id for better display.
    groups = {}
    for s, grp in assignment.items():
        groups.setdefault(grp, []).append(s)
    
    for grp, schools in groups.items():
        group_enrollment = sum(df.loc[s, "ENROLLMENT"] for s in schools)
        # Compute group's isp_count and check qualification: (I_g >= 0.25*E_g)
        group_isp = sum(df.loc[s, "isp_count"] for s in schools)
        qualifies = "Yes" if group_isp >= 0.25 * group_enrollment else "No"
        print(f"Group {grp}: Schools {schools} | Enrollment: {group_enrollment} | Qualifies: {qualifies}")

    # Compute comparison values.
    one_group_val, individual_val = compute_comparisons(df)
    print("\nComparison Benchmarks:")
    print(f"Total enrollment (if all schools in one group): {one_group_val}")
    print(f"Total enrollment (if each school in its own group): {individual_val}")

if __name__ == "__main__":
    main()