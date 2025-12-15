"""
Filter ID Helper Tool
Utility to convert between filter IDs and shift values.
"""

import argparse
from image_processor import ImageFilterProcessor


def print_all_filters():
    """Print all standard filter configurations."""
    HShiftList, VShiftList = ImageFilterProcessor.get_standard_hshift_vshift()
    
    print("=" * 80)
    print("Standard Filter Configurations")
    print("=" * 80)
    print(f"{'Filter ID':<12} {'HShift Value':<15} {'VShift Value':<15} {'Description':<30}")
    print("-" * 80)
    
    # Descriptions for some common filters
    descriptions = {
        1: "Normal vision",
        2: "Mild impairment",
        3: "Moderate impairment",
        4: "Severe impairment",
        5: "Very severe impairment",
        6: "Extreme impairment",
    }
    
    for i in range(len(HShiftList)):
        filter_id = i + 1
        hshift_raw = HShiftList[i]
        vshift = VShiftList[i]
        hshift = round(1 / hshift_raw, 4) if hshift_raw > 0 else 0
        
        desc = descriptions.get(filter_id, "")
        print(f"{filter_id:<12} {hshift:<15.4f} {vshift:<15.4f} {desc:<30}")
    
    print("=" * 80)


def get_filter_info(filter_id):
    """Get information for a specific filter ID."""
    try:
        hshift, vshift = ImageFilterProcessor.convert_filter_id_to_shifts(filter_id)
        
        print("=" * 60)
        print(f"Filter ID: {filter_id}")
        print("=" * 60)
        print(f"Horizontal Shift (hshift): {hshift}")
        print(f"Vertical Shift (vshift):   {vshift}")
        print("=" * 60)
        print("\nUsage in Python:")
        print(f"  hshift = {hshift}")
        print(f"  vshift = {vshift}")
        print("\nUsage in command line:")
        print(f"  --hshift {hshift} --vshift {vshift}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"Error: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Filter ID Helper - Convert between filter IDs and shift values'
    )
    
    parser.add_argument(
        '--filter_id', 
        type=int, 
        help='Filter ID to get shift values for'
    )
    parser.add_argument(
        '--list_all', 
        action='store_true',
        help='List all available filters'
    )
    
    args = parser.parse_args()
    
    if args.list_all:
        print_all_filters()
    elif args.filter_id:
        get_filter_info(args.filter_id)
    else:
        # Interactive mode
        print("Filter ID Helper - Interactive Mode")
        print("=" * 60)
        print("Options:")
        print("  1. Get shift values for a specific filter ID")
        print("  2. List all available filters")
        print("  3. Exit")
        print("=" * 60)
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == '1':
                    filter_id = int(input("Enter filter ID: ").strip())
                    print()
                    get_filter_info(filter_id)
                elif choice == '2':
                    print()
                    print_all_filters()
                elif choice == '3':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == '__main__':
    main()

