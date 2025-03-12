import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import time
import itertools
import os
from typing import List, Dict, Tuple, Set


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the input data for optimization.
    
    Args:
        df: Input DataFrame with school data
        
    Returns:
        Cleaned DataFrame with proper data types
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Handle column name variations
    column_mapping = {
        # Map possible column variations to our standard names
        'ISP': 'ISP',
        'ISP %': 'ISP',
        'ISP%': 'ISP',
        'Identified Student Percentage': 'ISP',
        
        'ENROLLMENT': 'ENROLLMENT',
        'Enrollment': 'ENROLLMENT',
        'Student Enrollment': 'ENROLLMENT',
        'Total Enrollment': 'ENROLLMENT',
        
        'LEA_NAME': 'LEA_NAME',
        'LEA Name': 'LEA_NAME',
        'District': 'LEA_NAME',
        'District Name': 'LEA_NAME',
        'School District': 'LEA_NAME',
        
        'SCHOOL': 'SCHOOL',
        'School': 'SCHOOL',
        'School Name': 'SCHOOL',
    }
    
    # Standardize column names (case-insensitive)
    for col in df.columns:
        for possible_name, standard_name in column_mapping.items():
            if col.upper() == possible_name.upper():
                df.rename(columns={col: standard_name}, inplace=True)
                break
    
    # Check if required columns exist
    required_columns = ['ISP', 'ENROLLMENT', 'SCHOOL', 'LEA_NAME']
    for col in required_columns:
        if col not in df.columns:
            # Try to find a close match
            close_matches = [c for c in df.columns if c.upper().replace(' ', '_') == col.upper()]
            if close_matches:
                df.rename(columns={close_matches[0]: col}, inplace=True)
            else:
                raise ValueError(f"Required column '{col}' not found in the data file. Available columns: {', '.join(df.columns)}")
    
    # Convert ISP from percentage string to float
    if isinstance(df['ISP'].iloc[0], str) and '%' in df['ISP'].iloc[0]:
        df['ISP'] = df['ISP'].str.rstrip('%').astype(float) / 100
    elif df['ISP'].dtype == 'float' or df['ISP'].dtype == 'int':
        # Check if the ISP is already in decimal form (less than 1) or percentage form (greater than 1)
        if df['ISP'].max() > 1:
            df['ISP'] = df['ISP'] / 100
    
    # Convert ENROLLMENT to integer
    df['ENROLLMENT'] = df['ENROLLMENT'].astype(int)
    
    # Create a unique school identifier
    df['SCHOOL_ID'] = df.index
    
    return df


def compute_reimbursement(isp: float) -> float:
    """
    Calculate reimbursement rate based on ISP value.
    
    Args:
        isp: The enrollment weighted ISP value (between 0 and 1)
        
    Returns:
        Reimbursement rate R
    """
    # Heaviside function implementation
    h = 1 if isp > 0.625 else 0
    
    # Continuous formula
    r = 4.5 * h + (4.5 * (isp * 1.6) + 0.5 * (1 - isp)) * (1 - h)
    return r


def calculate_group_metrics(schools: List[int], school_data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculate the weighted ISP and reimbursement for a group of schools.
    
    Args:
        schools: List of school IDs in the group
        school_data: DataFrame with school information
        
    Returns:
        Tuple of (weighted_isp, reimbursement_rate, total_enrollment)
    """
    group_df = school_data[school_data['SCHOOL_ID'].isin(schools)]
    total_enrollment = group_df['ENROLLMENT'].sum()
    
    if total_enrollment == 0:
        return 0, 0, 0
    
    # Calculate enrollment weighted ISP
    weighted_isp = sum(row['ISP'] * row['ENROLLMENT'] for _, row in group_df.iterrows()) / total_enrollment
    
    # Calculate reimbursement based on weighted ISP
    reimbursement = compute_reimbursement(weighted_isp)
    
    return weighted_isp, reimbursement, total_enrollment


def optimize_school_grouping(school_data: pd.DataFrame, max_groups: int = None) -> Tuple[List[List[int]], float]:
    """
    Optimize school groupings to maximize total weighted reimbursement.
    This uses a heuristic approach with incremental grouping.
    
    Args:
        school_data: DataFrame with school information
        max_groups: Maximum number of groups to consider (default: number of schools)
        
    Returns:
        Tuple of (list of school groups, total weighted reimbursement)
    """
    if max_groups is None:
        max_groups = len(school_data)
    
    # Start with each school in its own group
    school_ids = school_data['SCHOOL_ID'].tolist()
    
    # Initial configuration - each school in its own group
    current_groups = [[school_id] for school_id in school_ids]
    
    # Calculate initial metrics
    group_metrics = [calculate_group_metrics(group, school_data) for group in current_groups]
    total_enrollment = school_data['ENROLLMENT'].sum()
    
    # Calculate weighted reimbursement
    current_weighted_reimbursement = sum(metrics[1] * metrics[2] for metrics in group_metrics) / total_enrollment
    
    improved = True
    iteration = 0
    max_iterations = 1000  # Safeguard against infinite loops
    
    while improved and len(current_groups) > 1 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try merging each pair of groups
        best_merge = None
        best_improvement = 0
        
        for i, j in itertools.combinations(range(len(current_groups)), 2):
            # Merge groups i and j
            merged_group = current_groups[i] + current_groups[j]
            
            # Create a new group configuration after merge
            new_groups = [group for k, group in enumerate(current_groups) if k != i and k != j]
            new_groups.append(merged_group)
            
            # Calculate new metrics
            new_group_metrics = [calculate_group_metrics(group, school_data) for group in new_groups]
            new_weighted_reimbursement = sum(metrics[1] * metrics[2] for metrics in new_group_metrics) / total_enrollment
            
            # Check if this merge improves the total reimbursement
            improvement = new_weighted_reimbursement - current_weighted_reimbursement
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_merge = (i, j, new_groups, new_weighted_reimbursement)
        
        # If we found a beneficial merge, apply it
        if best_merge and best_improvement > 0:
            _, _, new_groups, new_weighted_reimbursement = best_merge
            current_groups = new_groups
            current_weighted_reimbursement = new_weighted_reimbursement
            improved = True
            
            # Break if we've reached the target number of groups
            if len(current_groups) <= max_groups:
                break
    
    return current_groups, current_weighted_reimbursement


def report_results(groups: List[List[int]], school_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed report of the optimized grouping.
    
    Args:
        groups: List of school groups (each group is a list of school IDs)
        school_data: DataFrame with school information
        
    Returns:
        DataFrame with group metrics
    """
    results = []
    total_enrollment = school_data['ENROLLMENT'].sum()
    
    # Calculate metrics for each group
    for i, group in enumerate(groups):
        weighted_isp, reimbursement_rate, group_enrollment = calculate_group_metrics(group, school_data)
        
        # Get school names in this group
        schools_in_group = school_data[school_data['SCHOOL_ID'].isin(group)]['SCHOOL'].tolist()
        schools_str = ", ".join(schools_in_group)
        
        # Calculate the contribution to overall weighted reimbursement
        contribution = (reimbursement_rate * group_enrollment) / total_enrollment
        
        results.append({
            'Group': i + 1,
            'Schools': schools_str,
            'Number of Schools': len(group),
            'Total Enrollment': group_enrollment,
            'Weighted ISP': f"{weighted_isp:.4f}",
            'Reimbursement Rate': f"{reimbursement_rate:.4f}",
            'Enrollment %': f"{(group_enrollment / total_enrollment) * 100:.2f}%",
            'Contribution to Total R': f"{contribution:.4f}"
        })
    
    # Calculate overall weighted reimbursement
    total_weighted_reimbursement = sum(float(r['Contribution to Total R']) for r in results)
    
    # Create a results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add a summary row
    summary = {
        'Group': 'TOTAL',
        'Schools': f"{len(school_data)} schools",
        'Number of Schools': len(school_data),
        'Total Enrollment': total_enrollment,
        'Weighted ISP': '',
        'Reimbursement Rate': f"{total_weighted_reimbursement:.4f}",
        'Enrollment %': '100.00%',
        'Contribution to Total R': f"{total_weighted_reimbursement:.4f}"
    }
    
    # Convert to DataFrame and append summary
    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)
    
    return results_df


def get_available_leas(input_file: str) -> List[str]:
    """
    Get a list of all unique LEA_NAME values from the input file.
    
    Args:
        input_file: Path to the input CSV or Excel file with school data
        
    Returns:
        List of unique LEA_NAME values
    """
    # Determine file extension and read accordingly
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Clean up column names to find the LEA_NAME column
    column_mapping = {
        'LEA_NAME': 'LEA_NAME',
        'LEA Name': 'LEA_NAME',
        'District': 'LEA_NAME',
        'District Name': 'LEA_NAME',
        'School District': 'LEA_NAME'
    }
    
    lea_col = None
    for col in df.columns:
        for possible_name, standard_name in column_mapping.items():
            if col.upper() == possible_name.upper():
                lea_col = col
                break
        if lea_col:
            break
    
    if not lea_col:
        # Try to find a close match
        close_matches = [c for c in df.columns if 'LEA' in c.upper() or 'DISTRICT' in c.upper()]
        if close_matches:
            lea_col = close_matches[0]
        else:
            raise ValueError(f"LEA_NAME or District column not found in the data file. Available columns: {', '.join(df.columns)}")
    
    # Return sorted list of unique LEA_NAME values
    return sorted(df[lea_col].unique().tolist())


def optimize_cep_groups(input_file: str, output_file: str = None, max_groups: int = None, lea_name: str = None):
    """
    Main function to run the optimization process from input file to results.
    
    Args:
        input_file: Path to the input CSV or Excel file with school data
        output_file: Path to save the output results (default: derived from input_file)
        max_groups: Maximum number of groups to consider (default: number of schools)
        lea_name: LEA_NAME to filter schools (default: None, uses all schools)
    """
    # Determine file extension and read accordingly
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Clean the data and standardize column names
    print("Cleaning data...")
    clean_df = clean_data(df)
    
    # Filter by LEA_NAME if specified
    if lea_name:
        if lea_name not in clean_df['LEA_NAME'].values:
            available_leas = clean_df['LEA_NAME'].unique().tolist()
            raise ValueError(f"LEA_NAME '{lea_name}' not found in dataset. Available LEA_NAME values: {available_leas}")
        
        print(f"Filtering data for LEA_NAME: {lea_name}")
        clean_df = clean_df[clean_df['LEA_NAME'] == lea_name]
        
        if clean_df.empty:
            raise ValueError(f"No schools found for LEA_NAME: {lea_name}")
    
    # Run optimization
    print(f"Optimizing groupings for {len(clean_df)} schools...")
    start_time = time.time()
    optimal_groups, total_r = optimize_school_grouping(clean_df, max_groups)
    end_time = time.time()
    
    # Generate report
    print("Generating results report...")
    results_df = report_results(optimal_groups, clean_df)
    
    # Save results if requested
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # Add LEA name to output file if filtering was applied
        if lea_name:
            safe_lea_name = "".join(c if c.isalnum() else "_" for c in lea_name)
            output_file = f"{base_name}_{safe_lea_name}_optimized_groups.xlsx"
        else:
            output_file = f"{base_name}_optimized_groups.xlsx"
    
    # Save to Excel with formatting
    print(f"Saving results to {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Optimized Groups', index=False)
        
        # Get the workbook and the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Optimized Groups']
        
        # Apply some formatting
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            worksheet.column_dimensions[column].width = adjusted_width
    
    # Print summary
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print(f"Number of groups: {len(optimal_groups)}")
    print(f"Overall weighted reimbursement rate: {total_r:.4f}")
    print(f"Results saved to {output_file}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize school groupings for CEP reimbursement')
    parser.add_argument('input_file', help='Path to input file (CSV or Excel)')
    parser.add_argument('--output', '-o', help='Path to output file (Excel)')
    parser.add_argument('--max-groups', '-m', type=int, help='Maximum number of groups')
    parser.add_argument('--lea-name', '-l', help='Filter by LEA_NAME (school district)')
    parser.add_argument('--list-leas', action='store_true', help='List all available LEA_NAME values and exit')
    
    args = parser.parse_args()
    
    # If --list-leas is specified, print all available LEA_NAME values and exit
    if args.list_leas:
        available_leas = get_available_leas(args.input_file)
        print("Available LEA_NAME values:")
        for lea in available_leas:
            print(f"  - {lea}")
        exit(0)
    
    optimize_cep_groups(args.input_file, args.output, args.max_groups, args.lea_name)
