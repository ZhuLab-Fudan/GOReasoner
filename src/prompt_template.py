"""
Prompt templates for GO term verification.
"""

from typing import Dict, List


class PromptTemplate:
    """
    Manages prompt templates for GOExplainer model.
    """
    
    def __init__(self):
        """Initialize prompt template."""
        self.system_prompt = (
            "You are an expert in protein biology and gene ontology annotation. "
            "Your task is to determine whether a given GO term accurately describes "
            "the protein based on provided evidence."
        )
        
        self.instruction = (
            "Based on the protein description and the candidate GO term, "
            "determine if the GO term is TRUE or FALSE for this protein. "
            "Provide your reasoning.\n\n"
        )
        
        self.input_template = (
            "**Protein Name**: {protein_name}\n"
            "**GO Domain**: {go_domain}\n\n"
            "**Protein Description**:\n{statement_text}\n\n"
            "**Candidate GO Term**:\n{candidates}\n\n"
            "Please respond in the following format:\n"
            "<GO_ID> <True/False> <Reasoning>"
        )
    
    def format_input(self,
                    protein_name: str,
                    go_domain: str,
                    statement_text: str,
                    candidates: str) -> str:
        """
        Format input text with protein and GO information.
        
        Args:
            protein_name: Protein identifier
            go_domain: GO domain (mf/bp/cc)
            statement_text: Protein description
            candidates: Formatted GO term candidates
            
        Returns:
            Formatted input string
        """
        return self.input_template.format(
            protein_name=protein_name,
            go_domain=go_domain,
            statement_text=statement_text,
            candidates=candidates
        )
    
    def create_messages(self, formatted_input: str) -> List[Dict[str, str]]:
        """
        Create chat messages for model input.
        
        Args:
            formatted_input: Formatted input text
            
        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.instruction + formatted_input}
        ]