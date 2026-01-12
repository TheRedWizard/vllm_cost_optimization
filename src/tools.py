"""
External data source tools with full auditability.

All tool calls are logged to the database with:
- Input parameters
- Output data
- Response time
- Status and errors
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import requests

try:
    from .db import Database
except ImportError:
    from db import Database


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry & Auditing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Result from a tool call."""
    success: bool
    data: Any
    error: str | None = None
    call_id: str | None = None
    response_time_ms: int | None = None


@dataclass
class ToolRegistry:
    """Registry of available tools with auditing."""
    
    db: Database | None = None
    _tools: dict[str, dict] = field(default_factory=dict)
    
    @classmethod
    def from_config(cls, config_path: str | Path = "infra/config.yaml") -> "ToolRegistry":
        """Create registry with database connection."""
        try:
            db = Database.from_config(config_path)
            db.connect()
        except Exception:
            db = None
        return cls(db=db)
    
    def register(
        self,
        name: str,
        category: str,
        description: str,
        parameters: dict[str, str],
    ) -> Callable:
        """Decorator to register a tool function."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> ToolResult:
                return self._execute_tool(name, category, func, args, kwargs)
            
            self._tools[name] = {
                "name": name,
                "category": category,
                "description": description,
                "parameters": parameters,
                "function": wrapper,
            }
            return wrapper
        return decorator
    
    def _execute_tool(
        self,
        name: str,
        category: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> ToolResult:
        """Execute a tool with auditing."""
        call_id = str(uuid.uuid4())
        input_params = {"args": list(args), "kwargs": kwargs}
        
        # Log start
        if self.db:
            try:
                self._log_tool_start(call_id, name, category, input_params)
            except Exception:
                pass
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Log success
            if self.db:
                try:
                    self._log_tool_success(call_id, result, elapsed_ms)
                except Exception:
                    pass
            
            return ToolResult(
                success=True,
                data=result,
                call_id=call_id,
                response_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            
            # Log error
            if self.db:
                try:
                    self._log_tool_error(call_id, error_msg, elapsed_ms)
                except Exception:
                    pass
            
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                call_id=call_id,
                response_time_ms=elapsed_ms,
            )
    
    def _log_tool_start(self, call_id: str, name: str, category: str, input_params: dict):
        with self.db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO tool_calls (call_id, tool_name, tool_category, input_params, status)
                VALUES (%s, %s, %s, %s, 'pending')
                """,
                (call_id, name, category, json.dumps(input_params))
            )
    
    def _log_tool_success(self, call_id: str, result: Any, elapsed_ms: int):
        # Create summary from result
        summary = self._summarize_result(result)
        
        with self.db.cursor() as cursor:
            cursor.execute(
                """
                UPDATE tool_calls SET
                    output_data = %s,
                    output_summary = %s,
                    response_time_ms = %s,
                    status = 'success',
                    completed_at = NOW(3)
                WHERE call_id = %s
                """,
                (json.dumps(result, default=str), summary[:500] if summary else None, elapsed_ms, call_id)
            )
    
    def _log_tool_error(self, call_id: str, error: str, elapsed_ms: int):
        with self.db.cursor() as cursor:
            cursor.execute(
                """
                UPDATE tool_calls SET
                    response_time_ms = %s,
                    status = 'error',
                    error_message = %s,
                    completed_at = NOW(3)
                WHERE call_id = %s
                """,
                (elapsed_ms, error, call_id)
            )
    
    def _summarize_result(self, result: Any) -> str:
        """Create a human-readable summary of the result."""
        if isinstance(result, list):
            return f"{len(result)} results"
        elif isinstance(result, dict):
            return f"dict with keys: {list(result.keys())[:5]}"
        elif isinstance(result, tuple):
            return f"tuple: {result[0] if result else 'empty'}..."
        return str(result)[:100]
    
    def get_tool(self, name: str) -> Callable | None:
        """Get a tool function by name."""
        tool = self._tools.get(name)
        return tool["function"] if tool else None
    
    def list_tools(self) -> list[dict]:
        """List all registered tools."""
        return [
            {
                "name": t["name"],
                "category": t["category"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in self._tools.values()
        ]
    
    def get_tools_for_llm(self) -> str:
        """Get tool descriptions formatted for LLM context."""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"\n## {t['name']} ({t['category']})")
            lines.append(f"{t['description']}")
            lines.append("Parameters:")
            for param, desc in t['parameters'].items():
                lines.append(f"  - {param}: {desc}")
        return "\n".join(lines)


# Create global registry
registry = ToolRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP Helper
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_USER_AGENT = "vllm-cost-optimization/1.0 (https://github.com/example; contact@example.com)"


def get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 30) -> Any:
    """Fetch JSON from a URL."""
    # Ensure User-Agent is set (required by Wikipedia, Wikidata, etc.)
    _headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        _headers.update(headers)
    r = requests.get(url, params=params, headers=_headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="wikipedia_search",
    category="wikipedia",
    description="Search Wikipedia articles by query",
    parameters={"query": "Search terms", "limit": "Max results (default 5)"},
)
def wikipedia_search(query: str, limit: int = 5) -> list[tuple[str, str]]:
    """Search Wikipedia and return (title, snippet) tuples."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    data = get_json(url, params=params)
    return [(x["title"], x["snippet"]) for x in data["query"]["search"]]


@registry.register(
    name="wikipedia_summary",
    category="wikipedia",
    description="Get summary of a Wikipedia article by title",
    parameters={"title": "Wikipedia article title"},
)
def wikipedia_summary(title: str) -> tuple[str, str]:
    """Get (title, extract) for a Wikipedia article."""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    data = get_json(url)
    return (data.get("title", ""), data.get("extract", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Wikidata
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="wikidata_search",
    category="wikidata",
    description="Search Wikidata entities",
    parameters={"query": "Search terms", "limit": "Max results (default 5)"},
)
def wikidata_search(query: str, limit: int = 5) -> list[tuple[str, str, str]]:
    """Search Wikidata entities, return (id, label, description) tuples."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": limit,
    }
    data = get_json(url, params=params)
    return [(x["id"], x["label"], x.get("description", "")) for x in data["search"]]


@registry.register(
    name="wikidata_sparql",
    category="wikidata",
    description="Run a SPARQL query against Wikidata",
    parameters={"query": "SPARQL query string"},
)
def wikidata_sparql(query: str) -> list[dict]:
    """Execute SPARQL query, return bindings."""
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    data = get_json(url, params={"query": query, "format": "json"}, headers=headers)
    return data["results"]["bindings"]


# ─────────────────────────────────────────────────────────────────────────────
# World Bank
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="worldbank_search_indicators",
    category="worldbank",
    description="Search World Bank indicators",
    parameters={"query": "Search terms", "limit": "Max results (default 5)"},
)
def worldbank_search_indicators(query: str, limit: int = 5) -> list[tuple[str, str]]:
    """Search World Bank indicators, return (id, name) tuples."""
    url = "https://api.worldbank.org/v2/indicator"
    params = {"format": "json", "per_page": 20000}
    data = requests.get(url, params=params, timeout=30).json()
    indicators = data[1] if len(data) > 1 else []
    hits = [i for i in indicators if query.lower() in (i.get("name", "").lower())]
    return [(h["id"], h["name"]) for h in hits[:limit]]


@registry.register(
    name="worldbank_country_indicator",
    category="worldbank",
    description="Get time series data for a country and indicator",
    parameters={
        "country_code": "Country code (e.g., USA)",
        "indicator": "Indicator ID (e.g., NY.GDP.PCAP.CD)",
        "per_page": "Number of data points (default 20)",
    },
)
def worldbank_country_indicator(country_code: str, indicator: str, per_page: int = 20) -> list[tuple[str, float]]:
    """Get (year, value) tuples for a country indicator."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {"format": "json", "per_page": per_page}
    data = requests.get(url, params=params, timeout=30).json()
    rows = data[1] if len(data) > 1 else []
    return [(r["date"], r["value"]) for r in rows if r["value"] is not None]


# ─────────────────────────────────────────────────────────────────────────────
# FRED (Federal Reserve Economic Data)
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="fred_search",
    category="fred",
    description="Search FRED economic data series (requires FRED_API_KEY env var)",
    parameters={"query": "Search terms", "limit": "Max results (default 5)"},
)
def fred_search(query: str, limit: int = 5) -> list[tuple[str, str]]:
    """Search FRED series, return (id, title) tuples."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set")
    url = "https://api.stlouisfed.org/fred/series/search"
    params = {"api_key": api_key, "file_type": "json", "search_text": query, "limit": limit}
    data = get_json(url, params=params)
    return [(s["id"], s["title"]) for s in data.get("seriess", [])]


@registry.register(
    name="fred_observations",
    category="fred",
    description="Get recent observations for a FRED series",
    parameters={"series_id": "Series ID (e.g., UNRATE)", "limit": "Number of observations (default 10)"},
)
def fred_observations(series_id: str, limit: int = 10) -> list[tuple[str, str]]:
    """Get (date, value) tuples for a FRED series."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "sort_order": "desc",
        "limit": limit,
    }
    data = get_json(url, params=params)
    return [(o["date"], o["value"]) for o in data.get("observations", [])]


# ─────────────────────────────────────────────────────────────────────────────
# US Census
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="census_acs_population",
    category="census",
    description="Get population for a US state (ACS 1-year)",
    parameters={"state_fips": "2-digit state FIPS code (e.g., 06 for California)"},
)
def census_acs_population(state_fips: str) -> dict[str, str]:
    """Get population data for a state."""
    url = "https://api.census.gov/data/2022/acs/acs1"
    params = {
        "get": "NAME,B01003_001E",
        "for": f"state:{state_fips}",
    }
    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    header, values = rows[0], rows[1]
    return dict(zip(header, values))


# ─────────────────────────────────────────────────────────────────────────────
# CoinGecko (Crypto)
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="coingecko_search",
    category="crypto",
    description="Search cryptocurrencies",
    parameters={"query": "Search terms"},
)
def coingecko_search(query: str) -> list[tuple[str, str, str]]:
    """Search coins, return (id, name, symbol) tuples."""
    url = "https://api.coingecko.com/api/v3/search"
    data = get_json(url, params={"query": query})
    return [(c["id"], c["name"], c["symbol"]) for c in data.get("coins", [])[:5]]


@registry.register(
    name="coingecko_price",
    category="crypto",
    description="Get current price for a cryptocurrency",
    parameters={"coin_id": "Coin ID from search", "vs": "Currency (default usd)"},
)
def coingecko_price(coin_id: str, vs: str = "usd") -> dict:
    """Get current price for a coin."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    data = get_json(url, params={"ids": coin_id, "vs_currencies": vs})
    return data.get(coin_id, {})


# ─────────────────────────────────────────────────────────────────────────────
# NOAA Weather (NWS)
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="weather_forecast",
    category="weather",
    description="Get weather forecast for a location (US only)",
    parameters={"lat": "Latitude", "lon": "Longitude"},
)
def weather_forecast(lat: float, lon: float) -> list[tuple[str, int, str, str]]:
    """Get forecast periods: (name, temp, unit, description)."""
    # Step 1: Get forecast URL from points endpoint
    points = get_json(f"https://api.weather.gov/points/{lat},{lon}")
    forecast_url = points["properties"]["forecast"]
    # Step 2: Fetch forecast
    forecast = get_json(forecast_url)
    periods = forecast["properties"]["periods"][:5]
    return [(p["name"], p["temperature"], p["temperatureUnit"], p["shortForecast"]) for p in periods]


# ─────────────────────────────────────────────────────────────────────────────
# NASA
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="nasa_apod",
    category="nasa",
    description="Get NASA Astronomy Picture of the Day for a date range",
    parameters={"start_date": "Start date (YYYY-MM-DD)", "end_date": "End date (YYYY-MM-DD)"},
)
def nasa_apod(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Get (date, title) tuples for APOD in date range."""
    api_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")
    url = "https://api.nasa.gov/planetary/apod"
    params = {"api_key": api_key, "start_date": start_date, "end_date": end_date}
    data = get_json(url, params=params)
    if isinstance(data, list):
        return [(x["date"], x["title"]) for x in data[:5]]
    return [(data["date"], data["title"])]


# ─────────────────────────────────────────────────────────────────────────────
# GDELT (News)
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="gdelt_news_search",
    category="news",
    description="Search recent news articles via GDELT",
    parameters={"query": "Search terms", "max_results": "Max articles (default 10)"},
)
def gdelt_news_search(query: str, max_results: int = 10) -> list[tuple[str, str, str]]:
    """Search news, return (date, title, url) tuples."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_results,
        "sort": "datedesc",
    }
    data = get_json(url, params=params)
    articles = data.get("articles", [])
    return [(a.get("seendate", ""), a.get("title", ""), a.get("url", "")) for a in articles]


# ─────────────────────────────────────────────────────────────────────────────
# arXiv (Papers)
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="arxiv_search",
    category="papers",
    description="Search arXiv papers",
    parameters={"query": "Search terms", "max_results": "Max papers (default 5)"},
)
def arxiv_search(query: str, max_results: int = 5) -> list[tuple[str, str]]:
    """Search arXiv, return (title, link) tuples."""
    try:
        import feedparser
    except ImportError:
        raise ImportError("feedparser not installed. Run: pip install feedparser")
    
    base = "http://export.arxiv.org/api/query"
    q = f"all:{query}"
    url = f"{base}?search_query={urllib.parse.quote(q)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    feed = feedparser.parse(url)
    return [(e.title.replace("\n", " "), e.link) for e in feed.entries]


# ─────────────────────────────────────────────────────────────────────────────
# PubMed (Medical/Bio Papers)
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="pubmed_search",
    category="papers",
    description="Search PubMed for medical/biological papers",
    parameters={"term": "Search terms", "max_results": "Max papers (default 5)"},
)
def pubmed_search(term: str, max_results: int = 5) -> list[str]:
    """Search PubMed, return list of PMIDs."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": max_results}
    data = get_json(url, params=params)
    return data["esearchresult"]["idlist"]


@registry.register(
    name="pubmed_summaries",
    category="papers",
    description="Get summaries for PubMed IDs",
    parameters={"pmids": "List of PubMed IDs"},
)
def pubmed_summaries(pmids: list[str]) -> list[tuple[str, str, str]]:
    """Get (pmid, title, pubdate) for given PMIDs."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
    data = get_json(url, params=params)
    out = []
    for pmid in pmids:
        item = data["result"].get(pmid, {})
        out.append((pmid, item.get("title", ""), item.get("pubdate", "")))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data.gov
# ─────────────────────────────────────────────────────────────────────────────


@registry.register(
    name="datagov_search",
    category="datasets",
    description="Search US government datasets on Data.gov",
    parameters={"query": "Search terms", "rows": "Max results (default 5)"},
)
def datagov_search(query: str, rows: int = 5) -> list[tuple[str, str]]:
    """Search Data.gov, return (title, notes) tuples."""
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": query, "rows": rows}
    data = get_json(url, params=params)
    results = data["result"]["results"]
    return [(r["title"], r.get("notes", "")[:200]) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution Helper
# ─────────────────────────────────────────────────────────────────────────────


def init_registry(config_path: str = "infra/config.yaml") -> ToolRegistry:
    """Initialize the global registry with database connection."""
    global registry
    # Connect DB to existing registry (keeps registered tools)
    try:
        db = Database.from_config(config_path)
        db.connect()
        registry.db = db
    except Exception:
        registry.db = None
    return registry


def call_tool(name: str, **kwargs) -> ToolResult:
    """Call a tool by name with given arguments."""
    tool = registry.get_tool(name)
    if not tool:
        return ToolResult(success=False, data=None, error=f"Unknown tool: {name}")
    return tool(**kwargs)


def list_tools() -> list[dict]:
    """List all available tools."""
    return registry.list_tools()
