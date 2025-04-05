import time
import random
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import DuckDuckGoSearchException

class SearchTool:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.last_search_time = 0
        self.min_delay = 5  # Minimum delay between searches (seconds)
        self.max_retries = 3  # Number of retries on rate limit errors

    def search_latest_news(self, query):
        current_time = time.time()
        
        # Enforce minimum delay between searches
        if current_time - self.last_search_time < self.min_delay:
            time.sleep(self.min_delay - (current_time - self.last_search_time))
        
        for attempt in range(self.max_retries + 1):
            try:
                # Perform the search
                results = self.search.run(f"{query} latest news")
                self.last_search_time = time.time()
                return results.split("\n")
            except DuckDuckGoSearchException as e:
                # Check for rate limit error (HTTP 429)
                if "429" in str(e) or "Ratelimit" in str(e):
                    # Wait exponentially longer before retrying
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                return f"Error: {str(e)}. Try again later."
        
        return "Failed to retrieve results due to rate limiting. Please try again later."