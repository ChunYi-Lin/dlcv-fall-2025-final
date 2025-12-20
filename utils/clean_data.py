import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Delete annotations with specific image from JSON file')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_file', help='Path to output JSON file')
    parser.add_argument('--image', default='034000.png', help='Image name to remove (default: 034000.png)')
    
    args = parser.parse_args()

    # Load the JSON data
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Filter out annotations with the specified image
    filtered_data = [item for item in data if item.get("image") != args.image]

    # Print how many were removed
    print(f"Original count: {len(data)}")
    print(f"Filtered count: {len(filtered_data)}")
    print(f"Removed: {len(data) - len(filtered_data)} annotations")

    # Save the filtered data
    with open(args.output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Saved to {args.output_file}")

if __name__ == '__main__':
    main()