import json
import re
import sys
import argparse
from typing import Dict, Any, Union, Optional


class JSONConverter:
    """
    A class to handle the conversion of raw text responses containing JSON data
    into valid JSON structures.
    """
    
    @staticmethod
    def extract_json(raw_text: str) -> Optional[str]:
        """
        Extracts JSON content from a raw text string using multiple strategies.
        
        Args:
            raw_text: String containing JSON data somewhere within it
            
        Returns:
            The extracted JSON string or None if no JSON is found
        """
        # Strategy 1: Look for JSON inside code blocks (```json { ... } ```)
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        match = re.search(code_block_pattern, raw_text, re.DOTALL)
        if match:
            return match.group(1)
            
        # Strategy 2: Look for JSON object with common fields
        # This is a heuristic that looks for objects with common JSON fields
        common_fields_pattern = r'(\{[\s\S]*?(?:"classification"|"entities"|"contacts"|"data"|"response"|"result")[\s\S]*?\})'
        match = re.search(common_fields_pattern, raw_text, re.DOTALL)
        if match:
            return match.group(1)
            
        # Strategy 3: Find the largest {...} block in the text
        # This is a fallback strategy that might work for simple cases
        json_blocks = re.findall(r'(\{[\s\S]*?\})', raw_text, re.DOTALL)
        if json_blocks:
            # Sort by length and return the largest one
            largest_block = max(json_blocks, key=len)
            return largest_block
            
        # Strategy 4: Look for array blocks
        array_blocks = re.findall(r'(\[[\s\S]*?\])', raw_text, re.DOTALL)
        if array_blocks:
            largest_array = max(array_blocks, key=len)
            return largest_array
            
        return None
    
    @staticmethod
    def fix_common_json_errors(json_str: str) -> str:
        """
        Fixes common JSON syntax errors.
        
        Args:
            json_str: JSON string that might contain syntax errors
            
        Returns:
            Corrected JSON string
        """
        # Remove trailing commas in objects
        json_str = re.sub(r',\s*}', '}', json_str)
        
        # Remove trailing commas in arrays
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unquoted keys (common error)
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix single quotes to double quotes (careful with this one)
        # This is tricky because we don't want to replace single quotes inside double-quoted strings
        # A more sophisticated approach might be needed
        json_str = re.sub(r"(?<![\\])(')([\w\s]+?)(?<![\\])(')\s*:", r'"\2":', json_str)
        
        # Replace single-quoted strings with double-quoted strings
        pattern = r"(?<![\\])(')(.*?)(?<![\\])(')"
        
        def replace_quotes(match):
            # Don't replace if inside a double-quoted string
            content = match.group(2)
            # Escape any double quotes in the content
            content = content.replace('"', '\\"')
            return f'"{content}"'
        
        # Apply selectively to avoid breaking JSON that's already valid
        try:
            json.loads(json_str)
            return json_str  # Already valid, don't mess with it
        except json.JSONDecodeError:
            # Only fix quotes if we have a JSON error
            json_str = re.sub(pattern, replace_quotes, json_str)
            return json_str
    
    @staticmethod
    def convert(raw_text: str) -> Dict[str, Any]:
        """
        Converts a raw text string containing JSON into a Python dictionary.
        
        Args:
            raw_text: Text string that contains JSON data
            
        Returns:
            Dictionary representation of the JSON data
            
        Raises:
            ValueError: If no valid JSON could be extracted or parsed
        """
        # Extract JSON string from the raw text
        json_str = JSONConverter.extract_json(raw_text)
        
        if not json_str:
            raise ValueError("No JSON content could be found in the raw text")
        
        # Try to parse it directly first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If direct parsing fails, try to fix common errors
            fixed_json_str = JSONConverter.fix_common_json_errors(json_str)
            
            try:
                return json.loads(fixed_json_str)
            except json.JSONDecodeError as e:
                # If all else fails, provide diagnostic information
                error_msg = (f"Failed to parse JSON even after fixes. Error: {str(e)}\n"
                             f"Extracted content: {json_str[:100]}...")
                raise ValueError(error_msg)


def format_json(json_data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format JSON data with specified indentation.
    
    Args:
        json_data: The data to format
        indent: Number of spaces to use for indentation
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(json_data, indent=indent, ensure_ascii=False)


def main():
    """Main function to run the converter from command line."""
    parser = argparse.ArgumentParser(description='Convert raw text with JSON to valid JSON')
    parser.add_argument('input', nargs='?', type=str, help='Input file path or raw JSON string')
    parser.add_argument('-o', '--output', type=str, help='Output file path')
    parser.add_argument('-i', '--indent', type=int, default=2, help='JSON indentation level')
    
    args = parser.parse_args()
    
    # Get the input text
    if args.input:
        try:
            # First try to read it as a file
            with open(args.input, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except (FileNotFoundError, IsADirectoryError):
            # If not a file, treat it as a raw string
            raw_text = args.input
    else:
        # If no input argument, read from stdin
        raw_text = sys.stdin.read()
    
    try:
        # Convert the raw text to JSON
        json_data = JSONConverter.convert(raw_text)
        
        # Format the JSON
        formatted_json = format_json(json_data, args.indent)
        
        # Output the JSON
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"JSON saved to {args.output}")
        else:
            print(formatted_json)
            
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()