# CEP School Grouping Optimizer

This tool optimizes the grouping of schools to maximize the student weighted reimbursement rate under the Community Eligibility Provision (CEP) program.

## Problem Overview

The optimization problem involves:
- Grouping a set of schools S into N groups
- Each school has an enrollment (k) and an Identified Student Percentage (ISP)
- All schools must be assigned to exactly one group
- The goal is to maximize the overall enrollment-weighted reimbursement rate

## How It Works

The implementation uses a greedy heuristic algorithm that:

1. Starts with each school in its own group
2. Iteratively merges groups when the merger increases the overall weighted reimbursement
3. Continues until no beneficial mergers remain or until reaching the target number of groups

The reimbursement formula for a group is:
- If weighted ISP > 62.5%: R = 4.5
- Otherwise: R = 4.5 × (ISP × 1.6) + 0.5 × (1 - ISP)

## Requirements

- Python 3.6+
- Required libraries:
  - pandas
  - numpy
  - ortools
  - openpyxl

Install the requirements with:
```
pip install pandas numpy ortools openpyxl
```

## Usage

### Command Line Interface

```bash
python cep_optimizer.py [input_file] --output [output_file] --max-groups [max_groups]
```

Arguments:
- `input_file`: Path to CSV or Excel file with school data
- `--output` or `-o`: Path to save the output Excel file (optional)
- `--max-groups` or `-m`: Maximum number of allowed groups (optional)

### Using the Runner Script

The simplest way to run the optimizer is:

```bash
python run_optimizer.py
```

This will run the optimizer on the example data located in the `data` directory.

### Input Data Format

The input file should be a CSV or Excel file with the following columns:
- STATE
- LEA_NAME
- SCHOOL
- ISP (as percentage, e.g., "40.6%")
- ISP_CATEGORY
- PARTICIPATION_IN_CEP_(Y OR BLANK)
- ENROLLMENT

### Output

The tool generates an Excel file with detailed information about the optimized grouping:
- Group assignments for each school
- Group metrics (weighted ISP, reimbursement rate)
- Overall weighted reimbursement rate

## Algorithm Details

The implementation uses a greedy approach rather than exact optimization due to the complex, non-linear nature of the problem. The algorithm:

1. Initially assigns each school to its own group
2. Evaluates all possible group pairs that can be merged
3. Selects the merger that produces the highest increase in the overall weighted reimbursement
4. Continues merging until no beneficial mergers remain

This heuristic approach efficiently handles large numbers of schools while providing near-optimal solutions in reasonable time.

## Limitations

- The algorithm is a heuristic and may not always find the global optimum
- The optimization is computationally intensive for very large datasets (thousands of schools)
- The input data format must match the expected schema

## Future Improvements

Potential enhancements to consider:
- Implement an exact optimization with integer programming for smaller datasets
- Add features to handle constraints like geographic proximity or administrative boundaries
- Improve performance with parallel processing for large datasets