import pandas as pd
from ortools.linear_solver import pywraplp

def read_and_prepare_data(csv_file):
    """
    Reads the CSV file and prepares the school data.
    Computes:
      - Converts ISP from percentage string to fraction.
      - Enrollment as numeric.
      - isp_count = round(ISP_fraction * ENROLLMENT)
    Returns a DataFrame.
    """
    df = pd.read_csv(csv_file)
    
    def isp_to_fraction(isp_str):
        try:
            return float(isp_str.strip().strip("%")) / 100.0
        except Exception:
            return 0.0
    df["ISP_fraction"] = df["ISP"].apply(isp_to_fraction)
    df["ENROLLMENT"] = pd.to_numeric(df["ENROLLMENT"], errors="coerce")
    df["isp_count"] = df.apply(lambda row: int(round(row["ISP_fraction"] * row["ENROLLMENT"])), axis=1)
    return df

def solve_optimization(df):
    """
    Solves the optimization problem using ORtools.
    - Each school (row in df) is assigned to exactly one group.
    - Maximum possible groups = number of schools.
    For each group g:
      Let E_g = sum(enrollment of schools in group g)
      Let I_g = sum(isp_count of schools in group g)
    A binary variable y[g] = 1 if the group qualifies (I_g >= 0.25 * E_g).
    Introduce variable z[g] which equals E_g when group qualifies, linearized as:
      z[g] <= E_g
      z[g] <= M * y[g]
      z[g] >= E_g - M*(1 - y[g])
    The objective is to maximize the sum of z[g] over all groups.
    """
    schools = df.index.tolist()
    n = len(schools)
    groups = list(range(n))  # Maximum groups equals the number of schools

    # Big-M: Maximum total enrollment (cast to float to avoid type issues)
    M = float(df["ENROLLMENT"].sum())

    # Create solver instance with CBC as the backend
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
    if not solver:
        print("Solver not available.")
        return None, None, None
    
    # Decision variables
    # x[s, g] = 1 if school s is assigned to group g.
    x = {}
    for s in schools:
        for g in groups:
            x[s, g] = solver.BoolVar(f"x_{s}_{g}")
    
    # y[g] = 1 if group g qualifies (I_g >= 0.25 * E_g)
    y = {}
    for g in groups:
        y[g] = solver.BoolVar(f"y_{g}")
    
    # z[g] represents group enrollment if the group qualifies, otherwise 0.
    z = {}
    for g in groups:
        z[g] = solver.NumVar(0, M, f"z_{g}")
    
    # Constraint: Each school must be assigned to exactly one group.
    for s in schools:
        solver.Add(sum(x[s, g] for g in groups) == 1)
    
    # For each group, define group enrollment (E_g) and group isp_count (I_g) as expressions.
    # Then add the qualification and linearization constraints.
    E_expr = {}
    I_expr = {}
    for g in groups:
        E_expr[g] = solver.Sum(df.loc[s, "ENROLLMENT"] * x[s, g] for s in schools)
        I_expr[g] = solver.Sum(df.loc[s, "isp_count"] * x[s, g] for s in schools)
        
        # Qualification constraint: if group qualifies (y[g]=1), then I_g >= 0.25 * E_g.
        # Using Big-M formulation: I_g >= 0.25 * E_g - M*(1-y[g])
        solver.Add(I_expr[g] >= 0.25 * E_expr[g] - M * (1 - y[g]))
        
        # Linearization constraints to link z[g] = E_expr[g] * y[g]:
        solver.Add(z[g] <= E_expr[g])
        solver.Add(z[g] <= M * y[g])
        solver.Add(z[g] >= E_expr[g] - M * (1 - y[g]))
    
    # Objective: maximize the sum of z[g] for all groups.
    objective = solver.Sum(z[g] for g in groups)
    solver.Maximize(objective)
    
    status = solver.Solve()
    
    assignment = {}
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # Extract assignment: For each school, assign it to the group where x[s, g] is 1.
        for s in schools:
            for g in groups:
                if x[s, g].solution_value() > 0.5:
                    assignment[s] = g
                    break
    else:
        print("No optimal solution found.")
        return None, None, solver

    solution_objective = solver.Objective().Value()
    return solution_objective, assignment, solver

def compute_comparisons(df):
    """
    Computes comparison benchmarks:
    1. All schools in one group.
    2. Each school in its own group.
    
    A group qualifies if sum(isp_count) >= 0.25 * sum(enrollment).
    """
    # 1. All schools in one group.
    total_enrollment = df["ENROLLMENT"].sum()
    total_isp_count = df["isp_count"].sum()
    qualifies_one_group = (total_isp_count >= 0.25 * total_enrollment)
    one_group_value = total_enrollment if qualifies_one_group else 0

    # 2. Each school in its own group:
    individual_value = 0
    for idx, row in df.iterrows():
        if row["isp_count"] >= 0.25 * row["ENROLLMENT"]:
            individual_value += row["ENROLLMENT"]
    return one_group_value, individual_value

def main():
    # Specify input CSV file (update path as needed)
    csv_file = "../gurobi/example.csv"
    df = read_and_prepare_data(csv_file)
    
    # Solve the optimization model.
    opt_value, assignment, solver = solve_optimization(df)
    if opt_value is None:
        return
    
    print("Optimization Results:")
    print(f"Optimal Objective (Total enrollment for qualified groups): {opt_value}")
    
    # Organize school assignments by group.
    groups_assignment = {}
    for s, grp in assignment.items():
        groups_assignment.setdefault(grp, []).append(s)
    
    for grp, schools in groups_assignment.items():
        group_enrollment = sum(df.loc[s, "ENROLLMENT"] for s in schools)
        group_isp = sum(df.loc[s, "isp_count"] for s in schools)
        qualifies = "Yes" if group_isp >= 0.25 * group_enrollment else "No"
        print(f"Group {grp}: Schools {schools} | Enrollment: {group_enrollment} | Qualifies: {qualifies}")
    
    # Compute comparison benchmarks.
    one_group_val, individual_val = compute_comparisons(df)
    print("\nComparison Benchmarks:")
    print(f"Total enrollment (if all schools in one group): {one_group_val}")
    print(f"Total enrollment (if each school in its own group): {individual_val}")

if __name__ == "__main__":
    main()
