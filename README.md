# Mixed integer linear programming Optimizations for CEP School nutrition programs

## Objective 1 Optimizing the number of students included in CEP qualifying groups

the model `student_coverage_optimization.py` groups schools in a way that  all schools qualify  
This was harder before the CEP percentage was lowered to 25%. it still may help some districts.

## Objective 2 optimizing Reimbursement rates
The model `reimbursement_optimization.py` groups schools to optimize the CEP payments

The  reimbursement rate formula is:

 ```{Python}
 def grouped_cep_reimbursement(ISP_Count: list, enrollment: list, free_rate: float=4.5, paid_rate: float=0.5) -> float:
    """Take a list of ISP_counts and a list of enrollments and returns the enrollment weighted reimbursement
    """
    itot = sum(ISP_count)
    etot = sum(enrollment)
    I = itot/etot
    if I >= 0.625:
        return etot * 4.5
    elseif I >= 0.25:
        return etot * ((free_rate * I * 1.6) + paid_rate * (1 - I))
    else:
        return etot * 0.5
```

This is modified to create a piece-wise linear optimization problem which can be more quickly solved.  The solver takes user input on the number of groups.  It's best to start with small er numbers first and see if they solve the problem 


## Notes

For some districts the optimization is trivial because they fall above or below thresholds. For example if every schools in a LAS has an ISP >62.5 any grouping or no grouping will be the same. For LAS ear the 25% and 62.5% cut points the optimization tools  can improve coverage or reimbursement.
