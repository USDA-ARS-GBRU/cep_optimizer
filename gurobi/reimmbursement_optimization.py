
import argparse
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum

# ------------------------------------------------------------------------
# Function: grouped_cep_reimbursement
# This replicates the reimbursement calculation given aggregated ISP and enrollment.
# free_rate and paid_rate are fixed as per the description.
# For a group:
#   - if I >= 0.625: reimbursement = etot * 4.5
#   - if 0.25 <= I < 0.625: reimbursement = etot * ((4.5 * I * 1.6) + 0.5 * (1-I))
#         which simplifies to: 0.5*etot + 6.7*itot    (since itot = I * etot)
#   - else (I < 0.25): reimbursement = etot * 0.5
# ------------------------------------------------------------------------
def grouped_cep_reimbursement(isp_counts, enrollments, free_rate=4.5, paid_rate=0.5):
    etot = sum(enrollments)
    itot = sum(isp_counts)
    if etot == 0:
        return 0.0
    I = itot / etot
    if I >= 0.625:
        return etot * 4.5
    elif I >= 0.25:
        return etot * ((free_rate * I * 1.6) + paid_rate * (1 - I))
    else:
        return etot * 0.5

# ------------------------------------------------------------------------
# Main script using command line arguments
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gurobi optimization model for grouping schools to maximize enrollment weighted reimbursement.")
    parser.add_argument("--inputfile", type=str, required=True,
                        help="Path to the CSV input file containing school data")
    parser.add_argument("--groups", type=int, required=True,
                        help="Number of groups to form")
    args = parser.parse_args()
    
    # Read the data from the specified inputfile.
    df = pd.read_csv(args.inputfile)
    
    # Process the ISP column: remove "%" and convert to fraction.
    df["ISP"] = df["ISP"].astype(str).str.replace("%", "").astype(float) / 100.0
    df["ENROLLMENT"] = pd.to_numeric(df["ENROLLMENT"])
    
    # Compute ISP_COUNT as the rounded value of ISP * ENROLLMENT.
    df["ISP_COUNT"] = df.apply(lambda row: int(round(row["ISP"] * row["ENROLLMENT"])), axis=1)
    
    # List of schools with attributes (using index as id)
    schools = list(df.index)
    enrollment = df["ENROLLMENT"].to_dict()
    isp_count = df["ISP_COUNT"].to_dict()
    
    # Number of groups from command line arguments.
    N = args.groups
        
    # Build the optimization model using gurobipy.
    model = Model("School_Grouping")
    
    # Decision variables: x[i, g] = 1 if school i is assigned to group g.
    x = {}
    for i in schools:
        for g in range(N):
            x[i, g] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{g}")
    
    # For each group, define aggregate enrollment E[g] and aggregate ISP_COUNT I[g].
    E = {}
    I_val = {}
    for g in range(N):
        E[g] = model.addVar(vtype=GRB.CONTINUOUS, name=f"E_{g}")
        I_val[g] = model.addVar(vtype=GRB.CONTINUOUS, name=f"I_{g}")
    
    # For each group, add binary variables for piecewise selection:
    # y1: group where I/E >= 0.625
    # y2: group where 0.25 <= I/E < 0.625
    # y3: group where I/E < 0.25
    y1 = {}
    y2 = {}
    y3 = {}
    for g in range(N):
        y1[g] = model.addVar(vtype=GRB.BINARY, name=f"y1_{g}")
        y2[g] = model.addVar(vtype=GRB.BINARY, name=f"y2_{g}")
        y3[g] = model.addVar(vtype=GRB.BINARY, name=f"y3_{g}")
        model.addConstr(y1[g] + y2[g] + y3[g] == 1, name=f"piecewise_sum_{g}")
    
    model.update()
    
    # Link aggregate variables with decision variables.
    for g in range(N):
        model.addConstr(E[g] == quicksum(enrollment[i]*x[i, g] for i in schools), name=f"enroll_agg_{g}")
        model.addConstr(I_val[g] == quicksum(isp_count[i]*x[i, g] for i in schools), name=f"isp_agg_{g}")
    
    # Each school must be assigned to exactly one group.
    for i in schools:
        model.addConstr(quicksum(x[i, g] for g in range(N)) == 1, name=f"assign_{i}")
    
    # Big-M for piecewise constraints.
    M = df["ENROLLMENT"].sum()
    
    # Add constraints for piecewise conditions.
    for g in range(N):
        model.addConstr(I_val[g] >= 0.625 * E[g] - M*(1 - y1[g]), name=f"piece1_lower_{g}")
        model.addConstr(I_val[g] >= 0.25 * E[g] - M*(1 - y2[g]), name=f"piece2_lower_{g}")
        model.addConstr(I_val[g] <= 0.625 * E[g] + M*(1 - y2[g]), name=f"piece2_upper_{g}")
        model.addConstr(I_val[g] <= 0.25 * E[g] + M*(1 - y3[g]), name=f"piece3_upper_{g}")
    
    # Define the objective function.
    obj_expr = 0
    for g in range(N):
        obj_expr += 4.5 * E[g] * y1[g] + (0.5 * E[g] + 6.7 * I_val[g]) * y2[g] + 0.5 * E[g] * y3[g]
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.update()
    
    print("Optimizing the grouping model...")
    model.optimize()
    
    # Output the results.
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
        print("\nOptimal Group Assignments:")
        for g in range(N):
            group_schools = [i for i in schools if x[i, g].x > 0.5]
            group_enroll = sum(enrollment[i] for i in group_schools)
            group_isp = sum(isp_count[i] for i in group_schools)
            if y1[g].x > 0.5:
                piece = "I/E >= 0.625"
                R_g = 4.5 * group_enroll
            elif y2[g].x > 0.5:
                piece = "0.25 <= I/E < 0.625"
                R_g = 0.5 * group_enroll + 6.7 * group_isp
            else:
                piece = "I/E < 0.25"
                R_g = 0.5 * group_enroll
            print(f"Group {g}: {len(group_schools)} schools, Enrollment={group_enroll:.2f}, ISP_count={group_isp}, Piece: {piece}, Reimbursement=${R_g:.2f}")
        print(f"\nTotal reimbursement (optimized): ${model.objVal:.2f}")
        
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
