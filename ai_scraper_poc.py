import os
import json
import logging
from typing import List, Optional
try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

import requests
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output Schema (Fallback to dict if pydantic is missing)
if HAS_PYDANTIC:
    class VolumeDiscount(BaseModel):
        min_quantity: int = Field(description="Minimum quantity required for this discount tier")
        max_quantity: Optional[int] = Field(description="Maximum quantity for this tier, if any")
        price_per_unit: float = Field(description="The price per unit at this discount tier")

    class CompetitorPriceData(BaseModel):
        component_name: str = Field(description="The exact name of the component as listed")
        brand: Optional[str] = Field(description="The manufacturer or brand if mentioned")
        sku_or_part_number: Optional[str] = Field(description="The SKU or part number")
        equivalent_components: List[str] = Field(description="List of equivalent, substitute, or cross-referenced competitor par numbers mentioned")
        base_price: float = Field(description="The price for a single unit")
        currency: str = Field(description="The currency code, e.g. USD")
        volume_discounts: List[VolumeDiscount] = Field(description="List of all volume-based pricing tiers explicitly mentioned")
        stock_status: Optional[str] = Field(description="Availability status like 'In Stock', 'Out of Stock', etc.")
else:
    CompetitorPriceData = None # type: ignore


class PricingAIScraper:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        if not use_mock:
            if not HAS_GENAI:
                logging.warning("google-generativeai is not installed. Falling back to mock data.")
                self.use_mock = True
            else:
                # Initialize the Gemini model for structured output
                self.api_key = os.environ.get("GEMINI_API_KEY")
                if not self.api_key:
                    logging.warning("GEMINI_API_KEY environment variable not found. Falling back to mock data.")
                    self.use_mock = True
                else:
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel('gemini-2.5-flash')

    def fetch_clean_text(self, url: str) -> str:
        """Fetches URL and returns clean text without HTML noise."""
        if not HAS_BS4:
            logging.warning("BeautifulSoup not installed. Just returning raw response text.")
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        try:
            # We use a brief timeout to not hang on bad URLs
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            if HAS_BS4:
                # Clean HTML using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header", "svg"]):
                    element.extract()
                return soup.get_text(separator=' | ', strip=True)
            else:
                return response.text
                
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
            return ""

    def extract_with_ai(self, text: str, url: str):
        """Uses Gemini to parse the unstructured text into our strict Pydantic model."""
        if self.use_mock:
            return self._get_mock_data(url)

        prompt = f"Analyze the following scraped text from a medical supply competitor product page ({url}). Extract the product pricing, exact equivalent components/cross-references, and any volume discount tiers."
        try:
            # Generate content enforcing the Pydantic schema
            response = self.model.generate_content(
                contents=[prompt, "HTML TEXT CONTENT:", text[:8000]], # Limiting char count for context window
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CompetitorPriceData,
                )
            )
            return json.loads(response.text)
            
        except Exception as e:
            logging.error(f"AI Extraction failed for {url}: {e}")
            return None

    def _get_mock_data(self, url: str) -> dict:
        """Fallback mock data for demonstration purposes."""
        logging.info("Using mock AI extraction...")
        # Simulating different outputs based on the URL
        if "syringe" in url.lower():
            return {
                "source_url": url,
                "component_name": "10mL Luer-Lok Syringe",
                "brand": "BD",
                "sku_or_part_number": "302995",
                "equivalent_components": ["Terumo SS-10L", "Covidien 1181000777"],
                "base_price": 0.25,
                "currency": "USD",
                "volume_discounts": [
                    {"min_quantity": 1, "max_quantity": 99, "price_per_unit": 0.25},
                    {"min_quantity": 100, "max_quantity": 499, "price_per_unit": 0.18},
                    {"min_quantity": 500, "max_quantity": None, "price_per_unit": 0.12}
                ],
                "stock_status": "In Stock - Ships Today"
            }
        else:
            return {
                "source_url": url,
                "component_name": "Standard ECG Electrodes",
                "brand": "3M",
                "sku_or_part_number": "2560",
                "equivalent_components": ["ConMed Cleartrace", "Vermed SilveRest"],
                "base_price": 0.15,
                "currency": "USD",
                "volume_discounts": [
                    {"min_quantity": 1, "max_quantity": 999, "price_per_unit": 0.15},
                    {"min_quantity": 1000, "max_quantity": 4999, "price_per_unit": 0.10},
                    {"min_quantity": 5000, "max_quantity": None, "price_per_unit": 0.08}
                ],
                "stock_status": "Low Stock"
            }

    def process_urls(self, urls: List[str]) -> List[dict]:
        results = []
        for url in urls:
            logging.info(f"Scraping {url}...")
            text = self.fetch_clean_text(url)
            
            if not text and not self.use_mock:
                logging.warning(f"No text recovered from {url}, skipping AI extraction.")
                continue
                
            data = self.extract_with_ai(text, url)
            if data:
                # Format to a standardized dict
                if isinstance(data, dict):
                     data['source_url'] = url
                     results.append(data)
                elif hasattr(data, 'model_dump'): # pydantic v2
                     item = data.model_dump()
                     item['source_url'] = url
                     results.append(item)
                else:
                     results.append({"source_url": url, "extracted": data})
        return results

if __name__ == "__main__":
    test_urls = [
        "https://www.medline.com/product/Standard-ECG-Electrodes/Electrodes",
        "https://www.mckesson.com/products/10mL-luer-lok-syringe"
    ]
    
    print("==================================================")
    print(" AI Competitor Pricing & Equivalents Scraper POC ")
    print("==================================================")
    
    # Initialize scraper (Set use_mock=False and export GEMINI_API_KEY to use real AI)
    scraper = PricingAIScraper(use_mock=True) 
    
    extracted_data = scraper.process_urls(test_urls)
    
    print("\n[+] Final Extracted Structured Data:")
    print(json.dumps(extracted_data, indent=2))
    
    print("\n" + "-"*50)
    print("Next Steps for Production Pipeline:")
    print("- Ensure dependencies are installed: `pip install requests beautifulsoup4 pydantic google-generativeai`")
    print("- Pass real competitor URLs into `process_urls()`.")
    print("- To run the true AI extraction, set 'GEMINI_API_KEY' environment variable and instantiate `PricingAIScraper(use_mock=False)`.")
    print("- Output can then be integrated directly with your predictive XGBoost model.")
