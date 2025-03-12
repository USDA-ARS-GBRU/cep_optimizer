#!/usr/bin/env python3
"""
Simple runner script for the CEP Optimizer
"""
import os
import sys
from cep_optimizer import optimize_cep_groups, get_available_leas

def list_available_leas(input_file):
    """
    List all available LEA_NAME values in the input file.
    
    Args:
        input_file: Path to the input file
    """
    try:
        leas = get_available_leas(input_file)
        
        print("\nAvailable LEA_NAME/District values:")
        for i, lea in enumerate(leas, 1):
            print(f"{i}. {lea}")
        print()
        
        return leas
    except Exception as e:
        print(f"Error listing LEA_NAME/District values: {e}")
        sys.exit(1)

def interactive_mode():
    """
    Run the optimizer in interactive mode, prompting the user for inputs.
    """
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Using current directory instead.")
        data_dir = os.path.dirname(__file__)
    
    # List available data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    if not data_files:
        print(f"No CSV or Excel files found in {data_dir}")
        sys.exit(1)
    
    print("\nAvailable data files:")
    for i, file in enumerate(data_files, 1):
        print(f"{i}. {file}")
    
    # Select input file
    while True:
        try:
            file_choice = int(input("\nSelect file number: "))
            if 1 <= file_choice <= len(data_files):
                selected_file = data_files[file_choice - 1]
                input_file = os.path.join(data_dir, selected_file)
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # List available LEA_NAME values
    leas = list_available_leas(input_file)
    
    # Ask if user wants to filter by LEA_NAME
    filter_by_lea = input("\nDo you want to filter by District/LEA? (y/n): ").lower().strip()
    
    lea_name = None
    if filter_by_lea == 'y':
        while True:
            try:
                lea_choice = int(input("Select District/LEA number (or 0 to skip filtering): "))
                if lea_choice == 0:
                    break
                if 1 <= lea_choice <= len(leas):
                    lea_name = leas[lea_choice - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    # Ask for max groups
    max_groups_input = input("\nEnter maximum number of groups (or press Enter for unlimited): ").strip()
    max_groups = int(max_groups_input) if max_groups_input else None
    
    # Output file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if lea_name:
        safe_lea_name = "".join(c if c.isalnum() else "_" for c in lea_name)
        output_file = os.path.join(data_dir, f"{base_name}_{safe_lea_name}_optimized_groups.xlsx")
    else:
        output_file = os.path.join(data_dir, f"{base_name}_optimized_groups.xlsx")
    
    # Confirm settings
    print("\nOptimization Settings:")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"District/LEA filter: {lea_name if lea_name else 'None (all schools)'}")
    print(f"Maximum groups: {max_groups if max_groups else 'Unlimited'}")
    
    confirm = input("\nProceed with optimization? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Optimization cancelled.")
        sys.exit(0)
    
    # Run the optimizer
    print("\nRunning CEP optimizer...")
    try:
        results = optimize_cep_groups(input_file, output_file, max_groups, lea_name)
        print("\nOptimization complete!")
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)

def main():
    """
    Main function for running the optimizer.
    """
    if len(sys.argv) > 1:
        # Command-line mode - use argparse in cep_optimizer.py
        import argparse
        from cep_optimizer import optimize_cep_groups, get_available_leas
        
        parser = argparse.ArgumentParser(description='Optimize school groupings for CEP reimbursement')
        parser.add_argument('input_file', nargs='?', help='Path to input file (CSV or Excel)')
        parser.add_argument('--output', '-o', help='Path to output file (Excel)')
        parser.add_argument('--max-groups', '-m', type=int, help='Maximum number of groups')
        parser.add_argument('--lea-name', '-l', help='Filter by District/LEA')
        parser.add_argument('--list-leas', action='store_true', help='List all available District/LEA values and exit')
        parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
        
        args = parser.parse_args()
        
        if args.interactive:
            interactive_mode()
            return
        
        if not args.input_file:
            parser.print_help()
            return
            
        # If --list-leas is specified, print all available LEA_NAME values and exit
        if args.list_leas:
            available_leas = get_available_leas(args.input_file)
            print("Available District/LEA values:")
            for lea in available_leas:
                print(f"  - {lea}")
            exit(0)
        
        optimize_cep_groups(args.input_file, args.output, args.max_groups, args.lea_name)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
