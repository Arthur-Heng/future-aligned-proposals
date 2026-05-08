"""
ArXiv paper download utility.

Downloads PDFs from arXiv given paper IDs or Semantic Scholar paper info.
Supports caching to avoid redundant downloads.
"""

import os
import re
import time
import requests
from typing import Optional, Dict, List
from pathlib import Path


class ArxivDownloader:
    """Downloads papers from arXiv with caching support."""
    
    ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
    ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
    
    def __init__(self, cache_dir: str = "data/arxiv"):
        """
        Initialize the ArXiv downloader.
        
        Args:
            cache_dir: Directory to cache downloaded PDFs
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_filename(self, title: str, max_length: int = 80) -> str:
        """Sanitize paper title for use as filename."""
        # Remove special characters
        sanitized = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Truncate
        return sanitized[:max_length]
    
    def _get_cache_path(self, arxiv_id: str, title: Optional[str] = None) -> Path:
        """Get the cache path for a paper."""
        # Clean arxiv_id (remove version suffix like v1, v2 and replace slashes)
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        clean_id = clean_id.replace('/', '_')  # Replace slashes with underscores

        if title:
            filename = f"{clean_id}_{self._sanitize_filename(title)}.pdf"
        else:
            filename = f"{clean_id}.pdf"

        return self.cache_dir / filename
    
    def is_cached(self, arxiv_id: str) -> bool:
        """Check if a paper is already cached."""
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        clean_id = clean_id.replace('/', '_')  # Replace slashes with underscores
        # Check for any file starting with this arxiv_id
        for f in self.cache_dir.glob(f"{clean_id}*.pdf"):
            if f.stat().st_size > 1000:  # Must be > 1KB to be valid
                return True
        return False
    
    def get_cached_path(self, arxiv_id: str) -> Optional[Path]:
        """Get the cached path if it exists."""
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        clean_id = clean_id.replace('/', '_')  # Replace slashes with underscores
        for f in self.cache_dir.glob(f"{clean_id}*.pdf"):
            if f.stat().st_size > 1000:
                return f
        return None
    
    def download(
        self,
        arxiv_id: str,
        title: Optional[str] = None,
        force: bool = False,
        timeout: int = 60
    ) -> Optional[Path]:
        """
        Download a paper from arXiv.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.00234" or "2301.00234v1")
            title: Optional paper title for the filename
            force: If True, re-download even if cached
            timeout: Request timeout in seconds
        
        Returns:
            Path to the downloaded PDF, or None if failed
        """
        # Check cache first
        if not force:
            cached = self.get_cached_path(arxiv_id)
            if cached:
                return cached
        
        # Clean arxiv_id
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Construct URL
        pdf_url = self.ARXIV_PDF_URL.format(arxiv_id=clean_id)
        
        # Get output path
        output_path = self._get_cache_path(clean_id, title)
        
        try:
            print(f"  Downloading: {pdf_url}")
            response = requests.get(pdf_url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
                print(f"  ⚠ Not a PDF: {content_type}")
                return None
            
            # Save to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = output_path.stat().st_size
            print(f"  ✓ Downloaded: {output_path.name} ({file_size / 1024:.1f} KB)")
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Download failed: {e}")
            return None
    
    def download_from_semantic_scholar(
        self,
        paper_info: Dict,
        force: bool = False
    ) -> Optional[Path]:
        """
        Download a paper using Semantic Scholar paper info.
        
        Args:
            paper_info: Dictionary with paper info (must have 'externalIds' or 'paperId')
            force: If True, re-download even if cached
        
        Returns:
            Path to the downloaded PDF, or None if not available on arXiv
        """
        # Try to get arXiv ID from external IDs
        external_ids = paper_info.get('externalIds', {})
        arxiv_id = None
        
        if external_ids:
            arxiv_id = external_ids.get('ArXiv')
        
        if not arxiv_id:
            # Try to extract from URL or other fields
            url = paper_info.get('url', '')
            if 'arxiv.org' in url:
                match = re.search(r'arxiv.org/abs/(\d+\.\d+)', url)
                if match:
                    arxiv_id = match.group(1)
        
        if not arxiv_id:
            title = paper_info.get('title', 'Unknown')
            print(f"  ⚠ No arXiv ID for: {title[:50]}...")
            return None
        
        title = paper_info.get('title')
        return self.download(arxiv_id, title=title, force=force)
    
    def download_batch(
        self,
        papers: List[Dict],
        delay: float = 1.0,
        force: bool = False
    ) -> Dict[str, Optional[Path]]:
        """
        Download multiple papers with rate limiting.
        
        Args:
            papers: List of paper info dictionaries
            delay: Delay between downloads in seconds
            force: If True, re-download even if cached
        
        Returns:
            Dictionary mapping paper_id to downloaded path (or None if failed)
        """
        results = {}
        
        for i, paper in enumerate(papers):
            paper_id = paper.get('paperId', paper.get('paper_id', f'paper_{i}'))
            title = paper.get('title', 'Unknown')
            
            print(f"\n[{i+1}/{len(papers)}] {title[:60]}...")
            
            path = self.download_from_semantic_scholar(paper, force=force)
            results[paper_id] = path
            
            if i < len(papers) - 1:
                time.sleep(delay)
        
        # Summary
        downloaded = sum(1 for p in results.values() if p is not None)
        print(f"\n✓ Downloaded {downloaded}/{len(papers)} papers")
        
        return results


def extract_arxiv_id_from_title(title: str) -> Optional[str]:
    """Try to extract arXiv ID from a paper title (if present)."""
    # Pattern like "arXiv:2301.00234" or "[2301.00234]"
    match = re.search(r'(?:arXiv:?)?(\d{4}\.\d{4,5})', title)
    return match.group(1) if match else None


if __name__ == "__main__":
    # Test the downloader
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.semantic_api import SemanticScholarAPI
    
    print("="*60)
    print("Testing ArXiv Downloader")
    print("="*60)
    
    # Initialize
    downloader = ArxivDownloader(cache_dir="data/arxiv/test")
    api = SemanticScholarAPI()
    
    # Test with a known paper
    test_title = "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    print(f"\nSearching for: {test_title}")
    
    paper = api.get_paper_by_title(test_title)
    if paper:
        print(f"Found: {paper['title']}")
        print(f"External IDs: {paper.get('externalIds', {})}")
        
        path = downloader.download_from_semantic_scholar(paper)
        if path:
            print(f"✓ Downloaded to: {path}")
        else:
            print("✗ Could not download")
    else:
        print("Paper not found")

