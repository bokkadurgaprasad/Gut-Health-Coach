"""
Ethical web scraping for gut health medical content
Respects robots.txt, rate limits, and terms of service
"""
import requests
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from datetime import datetime
import re
import hashlib

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EthicalWebScraper:
    """Ethical web scraper with robots.txt compliance"""

    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config["headers"])
        self.robots_cache = {}

    def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            if base_url not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(f"{base_url}/robots.txt")
                try:
                    rp.read()
                    self.robots_cache[base_url] = rp
                except:
                    self.robots_cache[base_url] = None

            rp = self.robots_cache[base_url]
            if rp is None:
                return True

            return rp.can_fetch(self.config["headers"]["User-Agent"], url)

        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse web page with error handling"""
        if not self.check_robots_txt(url):
            logger.warning(f"Robots.txt disallows scraping: {url}")
            return None

        for attempt in range(self.config["max_retries"]):
            try:
                logger.info(f"Fetching: {url} (attempt {attempt + 1})")

                response = self.session.get(url, timeout=self.config["timeout"])
                response.raise_for_status()

                time.sleep(self.config["delay_between_requests"])

                return BeautifulSoup(response.content, 'html.parser')

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(2 ** attempt)

        return None

class MayoClinicScraper:
    """Specialized scraper for Mayo Clinic content"""

    def __init__(self, scraper: EthicalWebScraper):
        self.scraper = scraper

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract structured content from Mayo Clinic pages"""
        content = {
            'url': url,
            'source': 'mayo_clinic',
            'title': '',
            'content': '',
            'symptoms': [],
            'causes': [],
            'treatment': [],
            'prevention': [],
            'scraped_at': datetime.now().isoformat()
        }

        # Extract title
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)

        # Extract main content sections
        main_content = soup.find('div', class_='content') or soup.find('main')
        if main_content:
            # Extract symptoms
            symptoms_section = main_content.find('h2', string=re.compile(r'Symptoms', re.I))
            if symptoms_section:
                symptoms_div = symptoms_section.find_next_sibling('div') or symptoms_section.find_next('div')
                if symptoms_div:
                    content['symptoms'] = [li.get_text(strip=True) for li in symptoms_div.find_all('li')]

            # Extract causes
            causes_section = main_content.find('h2', string=re.compile(r'Causes', re.I))
            if causes_section:
                causes_div = causes_section.find_next_sibling('div') or causes_section.find_next('div')
                if causes_div:
                    content['causes'] = [p.get_text(strip=True) for p in causes_div.find_all('p')]

            # Extract general content
            paragraphs = main_content.find_all('p')
            content['content'] = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        return content

    def scrape_articles(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple Mayo Clinic articles"""
        articles = []

        for url in tqdm(urls, desc="Scraping Mayo Clinic"):
            full_url = urljoin(DATA_SOURCES['mayo_clinic']['base_url'], url)
            soup = self.scraper.fetch_page(full_url)

            if soup:
                article = self.extract_content(soup, full_url)
                if article['content']:
                    articles.append(article)

        return articles

class HealthlineScraper:
    """Specialized scraper for Healthline content"""

    def __init__(self, scraper: EthicalWebScraper):
        self.scraper = scraper

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract structured content from Healthline pages"""
        content = {
            'url': url,
            'source': 'healthline',
            'title': '',
            'content': '',
            'key_points': [],
            'scraped_at': datetime.now().isoformat()
        }

        # Extract title
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)

        # Extract main content
        article_body = soup.find('div', class_='article-body') or soup.find('article')
        if article_body:
            # Extract key points/takeaways
            takeaways = article_body.find('div', class_='takeaways') or article_body.find('div', class_='key-takeaways')
            if takeaways:
                content['key_points'] = [li.get_text(strip=True) for li in takeaways.find_all('li')]

            # Extract paragraphs
            paragraphs = article_body.find_all('p')
            content['content'] = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        return content

    def scrape_articles(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple Healthline articles"""
        articles = []

        for url in tqdm(urls, desc="Scraping Healthline"):
            full_url = urljoin(DATA_SOURCES['healthline']['base_url'], url)
            soup = self.scraper.fetch_page(full_url)

            if soup:
                article = self.extract_content(soup, full_url)
                if article['content']:
                    articles.append(article)

        return articles

class NIDDKScraper:
    """Specialized scraper for NIDDK content"""

    def __init__(self, scraper: EthicalWebScraper):
        self.scraper = scraper

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract structured content from NIDDK pages"""
        content = {
            'url': url,
            'source': 'niddk',
            'title': '',
            'content': '',
            'overview': '',
            'scraped_at': datetime.now().isoformat()
        }

        # Extract title
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)

        # Extract main content
        main_content = soup.find('div', class_='main-content') or soup.find('main')
        if main_content:
            # Extract overview
            overview_section = main_content.find('div', class_='overview') or main_content.find('div', class_='summary')
            if overview_section:
                content['overview'] = overview_section.get_text(strip=True)

            # Extract all paragraphs
            paragraphs = main_content.find_all('p')
            content['content'] = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        return content

    def scrape_articles(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple NIDDK articles"""
        articles = []

        for url in tqdm(urls, desc="Scraping NIDDK"):
            full_url = urljoin(DATA_SOURCES['niddk']['base_url'], url)
            soup = self.scraper.fetch_page(full_url)

            if soup:
                article = self.extract_content(soup, full_url)
                if article['content']:
                    articles.append(article)

        return articles

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    return text.strip()

def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """Remove duplicate articles based on content hash"""
    seen_hashes = set()
    unique_articles = []

    for article in articles:
        content_hash = hashlib.md5(article['content'].encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_articles.append(article)

    return unique_articles

def main():
    """Main scraping function"""
    logger.info("Starting ethical web scraping for gut health content")

    # Initialize scraper
    scraper = EthicalWebScraper(SCRAPING_CONFIG)

    # Initialize specialized scrapers
    mayo_scraper = MayoClinicScraper(scraper)
    healthline_scraper = HealthlineScraper(scraper)
    niddk_scraper = NIDDKScraper(scraper)

    # Collect articles from all sources
    all_articles = []

    # Scrape Mayo Clinic
    mayo_articles = mayo_scraper.scrape_articles(DATA_SOURCES['mayo_clinic']['gut_health_urls'])
    all_articles.extend(mayo_articles)
    logger.info(f"Collected {len(mayo_articles)} articles from Mayo Clinic")

    # Scrape Healthline
    healthline_articles = healthline_scraper.scrape_articles(DATA_SOURCES['healthline']['gut_health_urls'])
    all_articles.extend(healthline_articles)
    logger.info(f"Collected {len(healthline_articles)} articles from Healthline")

    # Scrape NIDDK
    niddk_articles = niddk_scraper.scrape_articles(DATA_SOURCES['niddk']['gut_health_urls'])
    all_articles.extend(niddk_articles)
    logger.info(f"Collected {len(niddk_articles)} articles from NIDDK")

    # Clean and deduplicate
    for article in all_articles:
        article['content'] = clean_text(article['content'])

    unique_articles = deduplicate_articles(all_articles)
    logger.info(f"Deduplicated to {len(unique_articles)} unique articles")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = RAW_DATA_DIR / f"gut_health_raw_data_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_articles, f, indent=2, ensure_ascii=False)

    # Save CSV for human readability
    csv_path = RAW_DATA_DIR / f"gut_health_raw_data_{timestamp}.csv"
    df = pd.DataFrame(unique_articles)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved {len(unique_articles)} articles to {json_path}")
    logger.info(f"Saved CSV version to {csv_path}")

    # Summary statistics
    sources = [article['source'] for article in unique_articles]
    source_counts = {source: sources.count(source) for source in set(sources)}

    print("\nCollection Summary:")
    print(f"Total articles: {len(unique_articles)}")
    for source, count in source_counts.items():
        print(f"  {source}: {count} articles")

    return unique_articles

if __name__ == "__main__":
    main()
