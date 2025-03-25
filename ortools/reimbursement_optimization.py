import argparse
import pandas as pd
import math
from ortools.linear_solver import pywraplp

def grouped_cep_reimbursement(isp_counts, enrollments, free_rate=4.5, paid_rate=0.5):
    """
    Calculates group-level reimbursement using the aggregated ISP and enrollment totals.
    For a group:
      - if I/E >= 0.625: reimbursement = enrollment_total * free_rate
      - if 0.25 <= I/E < 0.625: reimbursement = 0.5*enrollment_total + 6.7*isp_total
      - if I/E < 0.25: reimbursement = enrollment_total * paid_rate
    """
    etot = sum(enrollments)
    itot = sum(isp_counts)
    if etot == 0:
        return 0.0
    I_ratio = itot / etot
    if I_ratio >= 0.625:
        return etot * free_rate
    elif I_ratio >= 0.25:
        return etot * ((free_rate * I_ratio * 1.6) + paid_rate * (1 - I_ratio))
    else:
        return etot * paid_rate

def main():
    parser = argparse.ArgumentParser(description="ORtools optimization model for grouping schools to maximize enrollment weighted reimbursement.")
    parser.add_argument("--inputfile", type=str, required=True,
                        help="Path to the CSV input file containing school data")
    parser.add_argument("--groups", type=int, required=True,
                        help="Number of groups to form")
    args = parser.parse_args()
    
    # Read CSV data
    df = pd.read_csv(args.inputfile)
    
    # Process ISP column: remove "%" and convert to fraction.
    df["ISP"] = df["ISP"].astype(str).str.replace("%", "").astype(float) / 100.0
    df["ENROLLMENT"] = pd.to_numeric(df["ENROLLMENT"])
    
    # Compute ISP_COUNT as rounded value of ISP * ENROLLMENT.
    df["ISP_COUNT"] = df.apply(lambda row: int(round(row["ISP"] * row["ENROLLMENT"])), axis=1)
    
    # Prepare data dictionaries and indices.
    schools = list(df.index)
    enrollment = df["ENROLLMENT"].to_dict()  # integer enrollment
    isp_count = df["ISP_COUNT"].to_dict()      # integer ISP counts
    N = args.groups
    
    # Big-M parameters (upper bounds)
    M_enroll = int(df["ENROLLMENT"].sum())      # maximum enrollment per group if all schools in one group
    M_isp = int(df["ISP_COUNT"].sum())            # maximum isp count among all groups

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
    if not solver:
        print("Solver not found.")
        return

    # Decision variables:
    # x[i,g] = 1 if school i is assigned to group g.
    x = {}
    for i in schools:
        for g in range(N):
            x[i, g] = solver.BoolVar(f"x_{i}_{g}")
    
    # Variables for aggregate enrollment and ISP count for group g.
    E = {}  # enrollment sum per group (integer)
    I_val = {}  # isp sum per group (integer)
    for g in range(N):
        E[g] = solver.NumVar(0, M_enroll, f"E_{g}")
        I_val[g] = solver.NumVar(0, M_isp, f"I_{g}")

    # Binary variables for piecewise selection per group.
    y1, y2, y3 = {}, {}, {}
    for g in range(N):
        y1[g] = solver.BoolVar(f"y1_{g}")  # I/E >= 0.625
        y2[g] = solver.BoolVar(f"y2_{g}")  # 0.25 <= I/E < 0.625
        y3[g] = solver.BoolVar(f"y3_{g}")  # I/E < 0.25
        # Each group must fall into exactly one piece.
        solver.Add(y1[g] + y2[g] + y3[g] == 1)
    
    # Define auxiliary variables to linearize the products E[g]*y and I_val[g]*y.
    # For group g: 
    # z1[g] = E[g]*y1[g], 
    # z2E[g] = E[g]*y2[g], 
    # z2I[g] = I_val[g]*y2[g],
    # z3[g] = E[g]*y3[g].
    z1, z2E, z2I, z3 = {}, {}, {}, {}
    for g in range(N):
        z1[g] = solver.NumVar(0, M_enroll, f"z1_{g}")
        z2E[g] = solver.NumVar(0, M_enroll, f"z2E_{g}")
        z2I[g] = solver.NumVar(0, M_isp, f"z2I_{g}")
        z3[g] = solver.NumVar(0, M_enroll, f"z3_{g}")
    
    # Set a sufficiently large constant for linearization (big-M)
    BIG = M_enroll  # for enrollment products; similarly, BIG_I for I_val will be M_isp.
    
    # Linearization constraints for products: z = E * y
    for g in range(N):
        # For z1[g] = E[g]*y1[g]
        solver.Add(z1[g] <= M_enroll * y1[g])
        solver.Add(z1[g] <= E[g])
        solver.Add(z1[g] >= E[g] - M_enroll * (1 - y1[g]))
        solver.Add(z1[g] >= 0)
        # For z2E[g] = E[g]*y2[g]
        solver.Add(z2E[g] <= M_enroll * y2[g])
        solver.Add(z2E[g] <= E[g])
        solver.Add(z2E[g] >= E[g] - M_enroll * (1 - y2[g]))
        solver.Add(z2E[g] >= 0)
        # For z2I[g] = I_val[g]*y2[g]
        solver.Add(z2I[g] <= M_isp * y2[g])
        solver.Add(z2I[g] <= I_val[g])
        solver.Add(z2I[g] >= I_val[g] - M_isp * (1 - y2[g]))
        solver.Add(z2I[g] >= 0)
        # For z3[g] = E[g]*y3[g]
        solver.Add(z3[g] <= M_enroll * y3[g])
        solver.Add(z3[g] <= E[g])
        solver.Add(z3[g] >= E[g] - M_enroll * (1 - y3[g]))
        solver.Add(z3[g] >= 0)
    
    # Link aggregate variables with decision variables.
    for g in range(N):
        solver.Add(E[g] == solver.Sum(enrollment[i] * x[i, g] for i in schools))
        solver.Add(I_val[g] == solver.Sum(isp_count[i] * x[i, g] for i in schools))
    
    # Each school is assigned to exactly one group.
    for i in schools:
        solver.Add(solver.Sum(x[i, g] for g in range(N)) == 1)
    
    # Add piecewise constraints using big-M formulation.
    # For y1: if selected then I_val[g] >= 0.625 * E[g].
    for g in range(N):
        solver.Add(I_val[g] >= 0.625 * E[g] - BIG * (1 - y1[g]))
    # For y2: if selected then 0.25*E[g] <= I_val[g] <= 0.625*E[g].
    for g in range(N):
        solver.Add(I_val[g] >= 0.25 * E[g] - BIG * (1 - y2[g]))
        solver.Add(I_val[g] <= 0.625 * E[g] + BIG * (1 - y2[g]))
    # For y3: if selected then I_val[g] <= 0.25 * E[g].
    for g in range(N):
        solver.Add(I_val[g] <= 0.25 * E[g] + BIG * (1 - y3[g]))
    
    # Objective: maximize total reimbursement.
    # Reimbursement per group:
    # If y1: reimbursement = free_rate * E[g] = 4.5 * E[g]
    # If y2: reimbursement = 0.5 * E[g] + 6.7 * I_val[g]
    # If y3: reimbursement = paid_rate * E[g] = 0.5 * E[g]
    # After linearization the bilinear terms become:
    #   for y1: 4.5 * (E[g]*y1) becomes 4.5 * z1[g]
    #   for y2: 0.5 * (E[g]*y2) becomes 0.5 * z2E[g] and 6.7 * (I_val[g]*y2) becomes 6.7 * z2I[g]
    #   for y3: 0.5 * (E[g]*y3) becomes 0.5 * z3[g]
    objective = solver.Sum(4.5 * z1[g] + (0.5 * z2E[g] + 6.7 * z2I[g]) + 0.5 * z3[g] for g in range(N))
    solver.Maximize(objective)
    
    print("Optimizing the grouping model with ORtools...")
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("\nOptimal Group Assignments:")
        total_obj = solver.Objective().Value()
        for g in range(N):
            # Retrieve schools in group g.
            group_schools = [i for i in schools if x[i, g].solution_value() > 0.5]
            group_enroll = sum(enrollment[i] for i in group_schools)
            group_isp = sum(isp_count[i] for i in group_schools)
            if y1[g].solution_value() > 0.5:
                piece = "I/E >= 0.625"
                R_g = 4.5 * group_enroll
            elif y2[g].solution_value() > 0.5:
                piece = "0.25 <= I/E < 0.625"
                R_g = 0.5 * group_enroll + 6.7 * group_isp
            else:
                piece = "I/E < 0.25"
                R_g = 0.5 * group_enroll
            print(f"Group {g}: {len(group_schools)} schools, Enrollment={group_enroll}, ISP_count={group_isp}, Piece: {piece}, Reimbursement=${R_g:.2f}")
        print(f"\nTotal reimbursement (optimized): ${total_obj:.2f}")
        
        total_enroll = df["ENROLLMENT"].sum()
        total_isp = df["ISP_COUNT"].sum()
        one_group_reimb = grouped_cep_reimbursement([total_isp], [total_enroll])
        print(f"Reimbursement if all schools are in one group: ${one_group_reimb:.2f}")
        
        individual_reimb = sum(grouped_cep_reimbursement([row["ISP_COUNT"]], [row["ENROLLMENT"]]) for _, row in df.iterrows())
        print(f"Reimbursement if each school is in its own group: ${individual_reimb:.2f}")
    else:
        print("No optimal solution found.")

if __name__ == '__main__':
    main()