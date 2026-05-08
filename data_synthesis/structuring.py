### Convert papers to a structured format

import json
import os
import re
import asyncio
from typing import Dict, Optional, Tuple, List
import pdfplumber
from tqdm import tqdm
from utils.api import call_chat_completion, calculate_cost, client

def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text content from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: all pages)
        
    Returns:
        Extracted text as a string
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            total_pages = len(pdf.pages)
            pages_to_read = min(max_pages, total_pages) if max_pages else total_pages
            
            print(f"📄 Extracting text from {pages_to_read} of {total_pages} pages...")
            
            for i in range(pages_to_read):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


def extract_paper_structure_with_upload(
    pdf_path: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> Tuple[Dict[str, str], float]:
    """
    Convert a research paper PDF into structured format by uploading to OpenAI API.
    
    This function uploads the PDF directly to OpenAI using the Assistants API,
    allowing OpenAI to process the PDF natively.
    
    Extracts the following fields (written in research proposal style):
    - research_question: The main research question(s) addressed
    - hypothesis: The hypothesis or hypotheses proposed
    - proposed_method: Detailed methodology
    - experiment_details: Experiments, datasets, baselines, metrics
    - novelty_claims: Claims about novelty and contributions
    
    Args:
        pdf_path: Path to the PDF file
        model: OpenAI model to use (default: gpt-4o, which supports vision/PDFs)
        temperature: Temperature for generation (default: 0.3 for more focused responses)
        max_tokens: Maximum tokens for response
        
    Returns:
        Tuple of (structured_data_dict, cost_in_usd)
    """
    import time
    
    file_id = None
    assistant = None
    
    try:
        # Upload PDF to OpenAI
        print(f"📤 Uploading PDF to OpenAI: {pdf_path}...")
        with open(pdf_path, "rb") as pdf_file:
            file_response = client.files.create(
                file=pdf_file,
                purpose="assistants"
            )
        
        file_id = file_response.id
        print(f"✅ File uploaded with ID: {file_id}")
        
        # Construct the instruction
        instruction = """You are an expert research paper analyzer. Your task is to extract structured information from research papers and rewrite it as a RESEARCH PROPOSAL (not a summary of an existing paper).

Extract the following information and return it in valid JSON format:

{
  "research_question": "The main research question(s). It should be broad and DOES NOT leak any key ideas.",
  "hypothesis": "The hypothesis or hypotheses (if any, otherwise 'Not specified')",
  "proposed_method": "A detailed description of the methodology, approach, algorithm, or method. Include key components, steps, and technical details. This should be 3-5 sentences minimum.",
  "experiment_details": "Description of the experiments, including datasets used, baselines compared, evaluation metrics, and key experimental setup details. This should be 2-4 sentences minimum.",
  "novelty_claims": "Claims about novelty, contributions, or innovations"
}

CRITICAL WRITING STYLE GUIDELINES:
- Write as a RESEARCH PROPOSAL, NOT as a summary of an existing paper
- NEVER use phrases like "The authors propose...", "This paper introduces...", "The paper presents...", "They develop...", "The authors show..."
- NEVER reference the paper as an external work (no "this paper", "the authors", "they", "the work")
- Instead, write in proposal style: describe what IS proposed, what WILL BE done, or use passive voice
- Good: "The proposed method introduces a novel framework..." or "A new approach is developed that..."
- Good: "The method achieves state-of-the-art results..." or "Experiments demonstrate that..."
- Bad: "The authors introduce ROBUSTALPACAEVAL..." or "This paper proposes a new benchmark..."
- Be DETAILED and comprehensive, especially for proposed_method and experiment_details
- If a field is not explicitly stated, infer from context when reasonable
- If information is truly not available, use "Not specified"

First give your reasoning process, then return the JSON response.
The json response should be:
```json
{
  "research_question": "...",
  "hypothesis": "...",
  "proposed_method": "...",
  "experiment_details": "...",
  "novelty_claims": "..."
}
```
"""

        # Create an assistant
        print(f"🤖 Creating assistant with model {model}...")
        assistant = client.beta.assistants.create(
            name="Paper Structure Extractor",
            instructions=instruction,
            model=model,
            tools=[{"type": "file_search"}],
            temperature=temperature,
        )
        
        # Create a thread
        print(f"📝 Creating thread...")
        thread = client.beta.threads.create()
        
        # Add message with file attachment
        print(f"💬 Adding message with file...")
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Analyze the uploaded research paper PDF and extract the structured information in the specified JSON format.",
            attachments=[{"file_id": file_id, "tools": [{"type": "file_search"}]}]
        )
        
        # Run the assistant
        print(f"▶️ Running assistant...")
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            max_completion_tokens=max_tokens,
        )
        
        # Wait for completion
        print(f"⏳ Waiting for completion...")
        while run.status in ["queued", "in_progress"]:
            time.sleep(2)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            print(f"   Status: {run.status}")
        
        if run.status != "completed":
            error_msg = f"Run failed with status: {run.status}"
            if hasattr(run, 'last_error') and run.last_error:
                error_msg += f"\nError details: {run.last_error}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Get the response
        print(f"📥 Retrieving response...")
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        response_text = messages.data[0].content[0].text.value
        
        print(f"📝 Raw response preview (first 500 chars):\n{response_text[:500]}")
        
        # Calculate cost (approximate)
        # Assistants API pricing is complex, this is a rough estimate
        cost = calculate_cost(model, run.usage.prompt_tokens, run.usage.completion_tokens) if run.usage else 0.0
        
        # Parse JSON response
        try:
            # Try to extract JSON if it's in markdown code blocks
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
                print(f"✓ Found JSON in markdown code block")
            else:
                # Try to find JSON object directly (non-greedy, find first complete object)
                # Look for a properly balanced JSON object
                start_idx = response_text.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = -1
                    for i in range(start_idx, len(response_text)):
                        if response_text[i] == '{':
                            brace_count += 1
                        elif response_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if end_idx != -1:
                        json_text = response_text[start_idx:end_idx]
                        print(f"✓ Found JSON object in response")
                    else:
                        print(f"⚠️ Could not find complete JSON object, trying full response")
                        json_text = response_text
                else:
                    print(f"⚠️ No JSON found, trying full response")
                    json_text = response_text
            
            print(f"📄 Extracted JSON (first 300 chars):\n{json_text[:300]}")
            structured_data = json.loads(json_text)
            print(f"✅ Successfully extracted structure. Cost: ${cost:.6f}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"📄 Full response text:\n{response_text}")
            print(f"📄 Attempted JSON extraction:\n{json_text if 'json_text' in locals() else 'N/A'}")
            raise ValueError(f"Invalid JSON response from API: {e}")
        
        return structured_data, cost
    
    except Exception as e:
        print(f"❌ Error during PDF upload and processing: {e}")
        raise
    finally:
        # Clean up resources (always executed)
        if assistant:
            try:
                client.beta.assistants.delete(assistant.id)
                print(f"🗑️  Deleted assistant {assistant.id}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete assistant: {e}")
        
        if file_id:
            try:
                client.files.delete(file_id)
                print(f"🗑️  Deleted uploaded file {file_id}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete uploaded file {file_id}: {e}")


def extract_paper_structure(
    pdf_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 3000,
    use_upload: bool = False,
) -> Tuple[Dict[str, str], float]:
    """
    Convert a research paper PDF into structured format using OpenAI API.
    
    Extracts the following fields (written in research proposal style):
    - research_question: The main research question(s) addressed
    - hypothesis: The hypothesis or hypotheses proposed
    - proposed_method: Detailed methodology (3-5 sentences)
    - experiment_details: Experiments, datasets, baselines, metrics (2-4 sentences)
    - novelty_claims: Claims about novelty and contributions
    
    Args:
        pdf_path: Path to the PDF file
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Temperature for generation (default: 0.3 for more focused responses)
        max_tokens: Maximum tokens for response
        use_upload: If True, upload PDF directly to OpenAI; if False, extract text locally
        
    Returns:
        Tuple of (structured_data_dict, cost_in_usd)
    """
    if use_upload:
        # Use file upload method (requires gpt-4o or similar model that supports PDFs)
        # if "gpt-4o" not in model and "gpt-5" not in model:
        #     print(f"⚠️ Warning: Model {model} may not support PDF uploads. Switching to gpt-4o")
        #     model = "gpt-4o"
        return extract_paper_structure_with_upload(
            pdf_path, 
            model=model, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
    
    # Extract text from PDF locally
    print(f"📄 Extracting text from {pdf_path}...")
    paper_text = extract_text_from_pdf(pdf_path)
    
    if not paper_text:
        raise ValueError("No text could be extracted from the PDF")
    
    # Truncate if too long (to avoid token limits)
    max_chars = 50000  # Roughly ~12.5k tokens
    if len(paper_text) > max_chars:
        print(f"⚠️ Paper text truncated from {len(paper_text)} to {max_chars} characters")
        paper_text = paper_text[:max_chars]
    
    # Construct the prompt
    system_prompt = """You are an expert research paper analyzer. Your task is to extract structured information from research papers and rewrite it as a RESEARCH PROPOSAL (not a summary of an existing paper).

Extract the following information and return it in valid JSON format:

{
  "research_question": "The main research question(s). It should be broad and DOES NOT leak any key ideas.",
  "hypothesis": "The hypothesis or hypotheses (if any, otherwise 'Not specified')",
  "proposed_method": "A detailed description of the methodology, approach, algorithm, or method. Include key components, steps, and technical details. This should be 3-5 sentences minimum.",
  "experiment_details": "Description of the experiments, including datasets used, baselines compared, evaluation metrics, and key experimental setup details. This should be 2-4 sentences minimum.",
  "novelty_claims": "Claims about novelty, contributions, or innovations"
}

CRITICAL WRITING STYLE GUIDELINES:
- Write as a RESEARCH PROPOSAL, NOT as a summary of an existing paper
- NEVER use phrases like "The authors propose...", "This paper introduces...", "The paper presents...", "They develop...", "The authors show..."
- NEVER reference the paper as an external work (no "this paper", "the authors", "they", "the work")
- Instead, write in proposal style: describe what IS proposed, what WILL BE done, or use passive voice
- Good: "The proposed method introduces a novel framework..." or "A new approach is developed that..."
- Good: "The method achieves state-of-the-art results..." or "Experiments demonstrate that..."
- Bad: "The authors introduce ROBUSTALPACAEVAL..." or "This paper proposes a new benchmark..."
- Be DETAILED and comprehensive, especially for proposed_method and experiment_details
- If a field is not explicitly stated, infer from context when reasonable
- If information is truly not available, use "Not specified"
- Return ONLY valid JSON, no additional text"""

    user_prompt = f"""Analyze the following research paper and extract the structured information:

{paper_text}

Return the structured information in the specified JSON format.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call OpenAI API
    print(f"🤖 Calling OpenAI API with model {model}...")
    response_text, cost = call_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}  # Ensure JSON response
    )
    
    # Parse JSON response
    try:
        structured_data = json.loads(response_text)
        print(f"✅ Successfully extracted structure. Cost: ${cost:.6f}")
        return structured_data, cost
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(f"Response was: {response_text}")
        raise ValueError(f"Invalid JSON response from API: {e}")


def process_paper_batch(
    pdf_paths: list,
    output_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_upload: bool = False,
) -> Tuple[list, float]:
    """
    Process multiple papers and optionally save results.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_path: Optional path to save results as JSON
        model: OpenAI model to use
        use_upload: If True, upload PDFs directly to OpenAI
        
    Returns:
        Tuple of (list of structured results, total cost)
    """
    results = []
    total_cost = 0.0
    
    # Create progress bar
    pbar = tqdm(pdf_paths, desc="Processing papers", unit="paper")
    
    for pdf_path in pbar:
        # Update progress bar description with current file
        pbar.set_postfix_str(f"Current: {os.path.basename(pdf_path)}")
        
        try:
            structured_data, cost = extract_paper_structure(
                pdf_path, 
                model=model,
                use_upload=use_upload,  
            )
            result = {
                "id": pdf_path.split("/")[-1].split(".")[0],
                "pdf_path": pdf_path,
                "status": "success",
                **structured_data
            }
            total_cost += cost
            pbar.set_postfix_str(f"✓ {os.path.basename(pdf_path)} | Cost: ${cost:.4f}")
        except Exception as e:
            pbar.write(f"❌ Error processing {pdf_path}: {e}")
            result = {
                "id": pdf_path.split("/")[-1].split(".")[0],
                "pdf_path": pdf_path,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results if output path specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {output_path}")
    
    print(f"\n💰 Total cost: ${total_cost:.6f}")
    print(f"📊 Processed {len(results)} papers ({sum(1 for r in results if r['status'] == 'success')} successful)")
    
    return results, total_cost


async def process_paper_async(
    pdf_path: str,
    model: str = "gpt-4o-mini",
    use_upload: bool = False,
    index: int = 0,
    total: int = 1,
) -> Dict:
    """
    Asynchronously process a single paper.
    
    Args:
        pdf_path: Path to the PDF file
        model: OpenAI model to use
        use_upload: If True, upload PDF directly to OpenAI
        index: Index of this paper in the batch (for logging)
        total: Total number of papers in the batch (for logging)
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Run the synchronous extraction in a thread pool
        structured_data, cost = await asyncio.to_thread(
            extract_paper_structure,
            pdf_path,
            model=model,
            use_upload=use_upload,
        )
        result = {
            "id": pdf_path.split("/")[-1].split(".")[0],
            "pdf_path": pdf_path,
            "status": "success",
            **structured_data
        }
        return result, cost
    except Exception as e:
        result = {
            "id": pdf_path.split("/")[-1].split(".")[0],
            "pdf_path": pdf_path,
            "status": "error",
            "error": str(e)
        }
        return result, 0.0


async def process_paper_batch_async(
    pdf_paths: List[str],
    output_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_upload: bool = False,
    max_concurrent: int = 5,
) -> Tuple[List[Dict], float]:
    """
    Process multiple papers asynchronously with concurrent requests.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_path: Optional path to save results as JSON
        model: OpenAI model to use
        use_upload: If True, upload PDFs directly to OpenAI
        max_concurrent: Maximum number of concurrent requests (default: 5)
        
    Returns:
        Tuple of (list of structured results, total cost)
    """
    print(f"\n🚀 Starting async batch processing of {len(pdf_paths)} papers")
    print(f"   Max concurrent requests: {max_concurrent}")
    print(f"   Model: {model}")
    print(f"   Upload mode: {'Enabled' if use_upload else 'Disabled (local extraction)'}\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(pdf_path: str, index: int) -> Tuple[Dict, float, int]:
        async with semaphore:
            result, cost = await process_paper_async(
                pdf_path,
                model=model,
                use_upload=use_upload,
                index=index,
                total=len(pdf_paths)
            )
            return result, cost, index
    
    # Process all papers concurrently (limited by semaphore)
    tasks = [
        process_with_semaphore(pdf_path, i)
        for i, pdf_path in enumerate(pdf_paths)
    ]
    
    # Create progress bar and wait for tasks to complete
    results_dict = {}
    total_cost = 0.0
    
    with tqdm(total=len(pdf_paths), desc="Processing papers (async)", unit="paper") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result, cost, index = await coro
                results_dict[index] = result
                total_cost += cost
                
                # Update progress bar
                filename = os.path.basename(result.get("pdf_path", ""))
                status = "✓" if result.get("status") == "success" else "✗"
                pbar.set_postfix_str(f"{status} {filename} | Cost: ${cost:.4f} | Total: ${total_cost:.2f}")
                pbar.update(1)
            except Exception as e:
                pbar.write(f"❌ Exception in task: {e}")
                pbar.update(1)
    
    # Sort results by original index to maintain order
    results = [results_dict.get(i, {
        "pdf_path": pdf_paths[i],
        "status": "error",
        "error": "Task failed to complete"
    }) for i in range(len(pdf_paths))]
    
    # Save results if output path specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {output_path}")
    
    print(f"\n💰 Total cost: ${total_cost:.6f}")
    print(f"📊 Processed {len(results)} papers ({sum(1 for r in results if r['status'] == 'success')} successful)")
    
    return results, total_cost


def run_async_batch(
    pdf_paths: List[str],
    output_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_upload: bool = False,
    max_concurrent: int = 5,
) -> Tuple[List[Dict], float]:
    """
    Synchronous wrapper for async batch processing.
    
    This function can be called from synchronous code and will run the async
    batch processing function.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_path: Optional path to save results as JSON
        model: OpenAI model to use
        use_upload: If True, upload PDFs directly to OpenAI
        max_concurrent: Maximum number of concurrent requests (default: 5)
        
    Returns:
        Tuple of (list of structured results, total cost)
    """
    return asyncio.run(
        process_paper_batch_async(
            pdf_paths=pdf_paths,
            output_path=output_path,
            model=model,
            use_upload=use_upload,
            max_concurrent=max_concurrent,
        )
    )


if __name__ == "__main__":
    # Example usage
    import sys
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Extract structured information from research papers in PDF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF
  python structuring.py path/to/paper.pdf -o output.json
  
  # Process multiple PDFs (batch)
  python structuring.py --batch path/to/pdfs/*.pdf -o results.json
  
  # Process with async (faster for multiple PDFs)
  python structuring.py --batch path/to/pdfs/*.pdf -o results.json --async --concurrent 5
  
  # Process with upload mode
  python structuring.py path/to/paper.pdf --upload --model gpt-4o
        """
    )
    parser.add_argument(
        "pdf_path", 
        nargs="*",
        help="Path to PDF file(s) or directory"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple PDFs (provide multiple paths or a directory)"
    )
    parser.add_argument(
        "--dir",
        help="Directory containing PDFs to process"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model", 
        default="gpt-4o-mini", 
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--upload", 
        action="store_true", 
        help="Upload PDF directly to OpenAI (requires gpt-4o or gpt-5)"
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async processing for batch mode (faster)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Maximum concurrent requests for async mode (default: 3)"
    )

    args = parser.parse_args()
    
    # Collect PDF paths
    pdf_paths = []
    
    if args.dir:
        # Process directory
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"❌ Error: Directory not found: {args.dir}")
            sys.exit(1)
        pdf_paths = sorted([str(p) for p in dir_path.glob("*.pdf")])
        if not pdf_paths:
            print(f"❌ Error: No PDF files found in {args.dir}")
            sys.exit(1)
        args.batch = True  # Auto-enable batch mode
    elif args.pdf_path:
        for path in args.pdf_path:
            p = Path(path)
            if p.is_dir():
                # If path is a directory, add all PDFs in it
                pdf_paths.extend(sorted([str(f) for f in p.glob("*.pdf")]))
                args.batch = True
            elif p.exists():
                pdf_paths.append(str(p))
            else:
                print(f"⚠️ Warning: File not found: {path}")
    else:
        parser.print_help()
        sys.exit(1)
    
    # Check if batch mode is needed
    if len(pdf_paths) > 1:
        args.batch = True
    
    # Process PDFs
    if args.batch:
        # Batch processing
        print("\n" + "="*60)
        print(f"BATCH PROCESSING: {len(pdf_paths)} PDFs")
        print("="*60)
        print(f"Mode: {'ASYNC' if args.use_async else 'SYNC'}")
        print(f"Model: {args.model}")
        print(f"Upload: {'Enabled' if args.upload else 'Disabled'}")
        if args.use_async:
            print(f"Max concurrent: {args.concurrent}")
        print("="*60 + "\n")
        
        if args.use_async:
            results, total_cost = run_async_batch(
                pdf_paths=pdf_paths,
                output_path=args.output,
                model=args.model,
                use_upload=args.upload,
                max_concurrent=args.concurrent,
            )
        else:
            results, total_cost = process_paper_batch(
                pdf_paths=pdf_paths,
                output_path=args.output,
                model=args.model,
                use_upload=args.upload,
            )
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total papers: {len(results)}")
        print(f"✓ Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"✗ Failed: {sum(1 for r in results if r['status'] == 'error')}")
        print(f"💰 Total cost: ${total_cost:.6f}")
        if results:
            print(f"📊 Avg cost/paper: ${total_cost/len(results):.6f}")
        if args.output:
            print(f"💾 Results saved to: {args.output}")
    else:
        # Single file processing
        pdf_path = pdf_paths[0]
        print("\n" + "="*60)
        print(f"Processing: {pdf_path}")
        if args.upload:
            print(f"Method: Upload to OpenAI (model: {args.model})")
        else:
            print(f"Method: Local text extraction (model: {args.model})")
        print("="*60 + "\n")
        
        structured_data, cost = extract_paper_structure(
            pdf_path,
            model=args.model,
            use_upload=args.upload,
        )
        
        # Print results
        print("\n" + "="*60)
        print("EXTRACTED STRUCTURE:")
        print("="*60)
        print(json.dumps(structured_data, indent=2))
        print(f"\n💰 Cost: ${cost:.6f}")
        
        # Save if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {args.output}")
