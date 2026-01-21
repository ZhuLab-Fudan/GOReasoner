"""
Convert test data from TSV format to JSON.
Reads a tab-separated file with protein, GO ID, and PubMed IDs,
and outputs structured JSON data for testing.

Input format (TSV):
    protein_id<TAB>goid<TAB>pmid1,pmid2,pmid3
    P12345<TAB>GO:0015631<TAB>12345678,23456789
    P12345<TAB>GO:0042802<TAB>12345678

Output format (JSON):
    [
        {
            "proid": "P12345",
            "go": [
                {
                    "goid": "GO:0015631",
                    "label": false
                },
                {
                    "goid": "GO:0042802",
                    "label": false
                }
            ],
            "desc": "",
            "pmids": ["12345678", "23456789"]
        }
    ]

Usage:
    python get_testdata.py \
        --input test_proteins.txt \
        --output testdata.json
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Set


class TestDataGenerator:
    """
    Convert test data from TSV to JSON format.
    """
    
    def __init__(self):
        """Initialize test data generator."""
        pass
    
    def parse_tsv_file(self, input_file: str) -> Dict[str, Dict]:
        """
        Parse TSV file and group by protein ID.
        
        Args:
            input_file: Path to input TSV file
            
        Returns:
            Dictionary mapping protein IDs to their GO terms and PMIDs
        """
        protein_data = defaultdict(lambda: {
            'goids': set(),
            'pmids': set(),
            'go_pmid_map': defaultdict(set)
        })
        
        print(f"Reading input file: {input_file}")
        
        line_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Parse TSV line
                parts = line.split('\t')
                
                if len(parts) < 2:
                    print(f"Warning: Line {line_num} has insufficient columns, skipping: {line}")
                    continue
                
                protein_id = parts[0].strip()
                goid = parts[1].strip()
                
                # Parse PMIDs (optional third column)
                pmids = []
                if len(parts) >= 3 and parts[2].strip():
                    pmid_str = parts[2].strip()
                    # Handle comma-separated or space-separated PMIDs
                    pmids = [p.strip() for p in pmid_str.replace(',', ' ').split() if p.strip()]
                
                # Validate GO ID format
                if not goid.startswith('GO:'):
                    print(f"Warning: Line {line_num} has invalid GO ID format: {goid}")
                    continue
                
                # Add to protein data
                protein_data[protein_id]['goids'].add(goid)
                protein_data[protein_id]['pmids'].update(pmids)
                
                # Map specific PMIDs to this GO term
                for pmid in pmids:
                    protein_data[protein_id]['go_pmid_map'][goid].add(pmid)
                
                line_count += 1
        
        print(f"Processed {line_count} lines")
        print(f"Found {len(protein_data)} unique proteins")
        
        return protein_data
    
    def convert_to_json_format(self, 
                              protein_data: Dict,
                              default_label: bool = False) -> List[Dict]:
        """
        Convert parsed data to JSON output format.
        
        Args:
            protein_data: Parsed protein data
            default_label: Default label value for GO terms
            
        Returns:
            List of protein dictionaries in output format
        """
        output_data = []
        
        for protein_id in sorted(protein_data.keys()):
            data = protein_data[protein_id]
            
            # Create GO term list with labels
            go_list = []
            for goid in sorted(data['goids']):
                go_list.append({
                    "goid": goid,
                    "label": default_label
                })
            
            # Create protein entry
            protein_entry = {
                "proid": protein_id,
                "go": go_list,
                "desc": "",  # Empty description by default
            }
            
            # Add PMIDs if available
            if data['pmids']:
                protein_entry["pmids"] = sorted(data['pmids'])
            
            output_data.append(protein_entry)
        
        return output_data
    
    def generate_test_data(self,
                          input_file: str,
                          output_file: str,
                          default_label: bool = False,
                          pretty_print: bool = True):
        """
        Main method to generate test data JSON from TSV.
        
        Args:
            input_file: Path to input TSV file
            output_file: Path to output JSON file
            default_label: Default label for GO terms
            pretty_print: Whether to format JSON with indentation
        """
        # Parse input file
        protein_data = self.parse_tsv_file(input_file)
        
        if not protein_data:
            print("Error: No valid data found in input file")
            sys.exit(1)
        
        # Convert to output format
        output_data = self.convert_to_json_format(
            protein_data, 
            default_label=default_label
        )
        
        # Write to JSON file
        print(f"\nWriting output to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            else:
                json.dump(output_data, f, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total proteins: {len(output_data)}")
        
        total_go_terms = sum(len(p['go']) for p in output_data)
        print(f"Total GO terms: {total_go_terms}")
        
        proteins_with_pmids = sum(1 for p in output_data if 'pmids' in p)
        print(f"Proteins with PMIDs: {proteins_with_pmids}")
        
        if proteins_with_pmids > 0:
            total_pmids = sum(len(p.get('pmids', [])) for p in output_data)
            print(f"Total unique PMIDs: {total_pmids}")
        
        print(f"{'='*60}\n")
        
        # Print sample entries
        print("Sample entries:")
        for i, protein in enumerate(output_data[:3], 1):
            print(f"\n{i}. Protein: {protein['proid']}")
            print(f"   GO terms: {len(protein['go'])}")
            if 'pmids' in protein:
                print(f"   PMIDs: {len(protein['pmids'])}")
        
        print(f"\nOutput saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert test data from TSV to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python get_testdata.py --input test_proteins.txt --output testdata.json
  
  # Set GO term labels to true
  python get_testdata.py --input test_proteins.txt --output testdata.json --label true
  
  # Compact JSON output
  python get_testdata.py --input test_proteins.txt --output testdata.json --no-pretty

Input file format (TSV):
  protein_id<TAB>goid<TAB>pmid1,pmid2
  P12345<TAB>GO:0015631<TAB>12345678,23456789
  P12345<TAB>GO:0042802<TAB>12345678
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input TSV file (protein<TAB>goid<TAB>pmids)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file'
    )
    
    parser.add_argument(
        '--label',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Default label value for GO terms (true/false, default: false)'
    )
    
    parser.add_argument(
        '--no-pretty',
        action='store_true',
        help='Disable pretty-printing (compact JSON)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    import os
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate test data
    generator = TestDataGenerator()
    generator.generate_test_data(
        input_file=args.input,
        output_file=args.output,
        default_label=args.label,
        pretty_print=not args.no_pretty
    )


if __name__ == "__main__":
    main()