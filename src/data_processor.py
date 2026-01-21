"""
Data processor for protein sequence and GO term candidates.
Handles chunk extraction from literature and data preprocessing.
"""

import json
import os
from typing import List, Dict, Optional
from tqdm import tqdm


class ProteinDataProcessor:
    """
    Process protein data including literature chunks and GO term candidates.
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50,
                 min_chunk_length: int = 100):
        """
        Initialize data processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Overlap tokens between chunks
            min_chunk_length: Minimum valid chunk length
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
    
    def get_chunks(self, 
                   text: str, 
                   protein_id: str) -> List[Dict[str, str]]:
        """
        Extract text chunks from literature with sliding window.
        
        Args:
            text: Full literature text
            protein_id: Protein identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        words = text.split()
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append({
                    'protein_id': protein_id,
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            start += self.chunk_size - self.overlap
        
        return chunks
    
    def process_literature_file(self, 
                               file_path: str,
                               output_path: Optional[str] = None) -> List[Dict]:
        """
        Process literature file and extract chunks for all proteins.
        
        Args:
            file_path: Path to input literature file
            output_path: Optional path to save processed chunks
            
        Returns:
            List of processed protein data with chunks
        """
        processed_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        for item in tqdm(raw_data, desc="Processing literature"):
            protein_id = item.get('proid')
            literature = item.get('desc', '')
            
            if len(literature) < self.min_chunk_length:
                continue
            
            chunks = self.get_chunks(literature, protein_id)
            
            processed_data.append({
                'proid': protein_id,
                'chunks': chunks,
                'metadata': {
                    'total_chunks': len(chunks),
                    'original_length': len(literature)
                }
            })
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        return processed_data
    
    def merge_with_candidates(self,
                            chunks_data: List[Dict],
                            candidates_file: str,
                            output_path: str) -> List[Dict]:
        """
        Merge literature chunks with GO term candidates.
        
        Args:
            chunks_data: Processed chunks data
            candidates_file: Path to GO candidates file
            output_path: Path to save merged data
            
        Returns:
            Merged dataset ready for inference
        """
        # Load GO candidates
        with open(candidates_file, 'r') as f:
            candidates = json.load(f)
        
        candidate_dict = {item['proid']: item['goids'] for item in candidates}
        
        merged_data = []
        for protein_data in chunks_data:
            proid = protein_data['proid']
            
            if proid in candidate_dict:
                merged_data.append({
                    'proid': proid,
                    'chunks': protein_data['chunks'],
                    'goids': candidate_dict[proid],
                    'metadata': protein_data['metadata']
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        return merged_data