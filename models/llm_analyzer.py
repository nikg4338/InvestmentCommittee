"""
LLM analyzer module using Google Gemini API
Uses Gemini to analyze macro conditions, news sentiment, and volatility risks
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from datetime import datetime, timedelta

# Handle Google API imports gracefully
try:
    import google.generativeai as genai  # type: ignore
    from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
    from google.api_core.exceptions import ResourceExhausted, GoogleAPIError  # type: ignore
    GOOGLE_AI_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Google AI packages not available: {e}")
    logger.error("Please install with: pip install google-generativeai")
    GOOGLE_AI_AVAILABLE = False
    
    # Create dummy classes for type hints
    class HarmCategory:
        HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
        HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
        HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    
    class HarmBlockThreshold:
        BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    
    class ResourceExhausted(Exception):
        pass
    
    class GoogleAPIError(Exception):
        pass

from config.settings import get_gemini_config, validate_gemini_api_key

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    MACRO_ANALYSIS = "macro_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_SUMMARY = "market_summary"

@dataclass
class AnalysisResult:
    """Result of LLM analysis"""
    analysis_type: AnalysisType
    content: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None

class RateLimiter:
    """Rate limiter for Gemini API calls with exponential backoff"""
    
    def __init__(self, requests_per_minute: int = 15, requests_per_day: int = 1500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = []
        self.day_requests = []
        self.last_request_time = 0
        self.backoff_time = 1  # Start with 1 second backoff
        self.max_backoff_time = 300  # Maximum 5 minutes backoff
        
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(current_time)
        
        # Check daily limit
        if len(self.day_requests) >= self.requests_per_day:
            wait_time = 86400 - (current_time - self.day_requests[0])
            if wait_time > 0:
                logger.warning(f"Daily rate limit reached. Waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self._clean_old_requests(time.time())
        
        # Check minute limit
        if len(self.minute_requests) >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.minute_requests[0])
            if wait_time > 0:
                logger.warning(f"Minute rate limit reached. Waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self._clean_old_requests(time.time())
        
        # Minimum time between requests (avoid rapid-fire requests)
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1:  # Minimum 1 second between requests
            time.sleep(1 - time_since_last)
        
        # Record this request
        current_time = time.time()
        self.minute_requests.append(current_time)
        self.day_requests.append(current_time)
        self.last_request_time = current_time
        
    def _clean_old_requests(self, current_time: float):
        """Remove old request timestamps"""
        # Remove minute requests older than 60 seconds
        self.minute_requests = [req for req in self.minute_requests 
                                if current_time - req < 60]
        
        # Remove day requests older than 24 hours
        self.day_requests = [req for req in self.day_requests 
                             if current_time - req < 86400]
    
    def handle_rate_limit_error(self):
        """Handle rate limit error with exponential backoff"""
        wait_time = min(self.backoff_time, self.max_backoff_time)
        logger.warning(f"Rate limit hit. Backing off for {wait_time} seconds")
        time.sleep(wait_time)
        self.backoff_time = min(self.backoff_time * 2, self.max_backoff_time)
    
    def reset_backoff(self):
        """Reset backoff time after successful request"""
        self.backoff_time = 1

class GeminiAnalyzer:
    """Gemini API client for investment analysis"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize Gemini analyzer"""
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google AI packages not available. Please install with: pip install google-generativeai")
        
        if not validate_gemini_api_key():
            raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY in .env file")
        
        config = get_gemini_config()
        genai.configure(api_key=config["api_key"])
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.rate_limiter = RateLimiter()
        
        # Safety settings for financial analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        logger.info(f"Initialized Gemini analyzer with model: {model_name}")
    
    def _make_request_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Make request to Gemini with retry logic and rate limiting"""
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Make the request
                response = self.model.generate_content(
                    prompt,
                    safety_settings=self.safety_settings,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,  # Lower temperature for more consistent financial analysis
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=2048,
                    )
                )
                
                # Check if response was blocked
                if response.candidates[0].finish_reason.name == "SAFETY":
                    logger.warning("Response was blocked by safety filters")
                    return "Response blocked by safety filters. Please rephrase your request."
                
                # Reset backoff on success
                self.rate_limiter.reset_backoff()
                
                return response.text
                
            except ResourceExhausted as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}): {e}")
                self.rate_limiter.handle_rate_limit_error()
                
                if attempt == max_retries - 1:
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts. Please try again later.")
                    
            except GoogleAPIError as e:
                logger.error(f"Google API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Google API error after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Unexpected error after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
        
        raise Exception("Failed to make request after all retries")
    
    def analyze_macro_conditions(self, economic_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze macro economic conditions"""
        start_time = time.time()
        
        prompt = f"""
        As an expert macroeconomic analyst, analyze the following economic data and provide insights for investment decisions:

        Economic Data:
        {json.dumps(economic_data, indent=2)}

        Please provide:
        1. Current macro environment assessment (bullish/bearish/neutral)
        2. Key economic indicators analysis
        3. Inflation and interest rate outlook
        4. Impact on equity markets
        5. Risk factors to monitor
        6. Investment recommendations based on current conditions

        Format your response as a structured analysis with clear sections.
        Provide a confidence score (0-100) for your assessment.
        """
        
        try:
            response = self._make_request_with_retry(prompt)
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.MACRO_ANALYSIS,
                content=response,
                confidence=0.85,  # This could be extracted from the response
                timestamp=datetime.now(),
                metadata={"economic_data": economic_data},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in macro analysis: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.MACRO_ANALYSIS,
                content=f"Error analyzing macro conditions: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> AnalysisResult:
        """Analyze market sentiment from news"""
        start_time = time.time()
        
        # Limit news articles to avoid token limits
        news_sample = news_data[:10] if len(news_data) > 10 else news_data
        
        prompt = f"""
        As a financial sentiment analyst, analyze the following news articles and provide market sentiment insights:

        News Articles:
        {json.dumps(news_sample, indent=2)}

        Please provide:
        1. Overall market sentiment (very bearish/bearish/neutral/bullish/very bullish)
        2. Key themes and trends identified
        3. Sentiment by sector (if applicable)
        4. Potential market impact
        5. Risk sentiment analysis
        6. Confidence score (0-100) for your sentiment assessment

        Focus on actionable insights for trading decisions.
        """
        
        try:
            response = self._make_request_with_retry(prompt)
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
                content=response,
                confidence=0.80,
                timestamp=datetime.now(),
                metadata={"news_count": len(news_sample)},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
                content=f"Error analyzing sentiment: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def analyze_volatility_risks(self, volatility_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze volatility and risk factors"""
        start_time = time.time()
        
        prompt = f"""
        As a risk management specialist, analyze the following volatility data and provide risk assessment:

        Volatility Data:
        {json.dumps(volatility_data, indent=2)}

        Please provide:
        1. Current volatility regime assessment
        2. Volatility trend analysis
        3. Risk-adjusted return outlook
        4. Volatility trading opportunities
        5. Risk management recommendations
        6. Position sizing suggestions based on current volatility
        7. Confidence score (0-100) for your risk assessment

        Focus on practical risk management insights.
        """
        
        try:
            response = self._make_request_with_retry(prompt)
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.VOLATILITY_ANALYSIS,
                content=response,
                confidence=0.75,
                timestamp=datetime.now(),
                metadata={"volatility_data": volatility_data},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.VOLATILITY_ANALYSIS,
                content=f"Error analyzing volatility: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def generate_market_summary(self, market_data: Dict[str, Any]) -> AnalysisResult:
        """Generate comprehensive market summary"""
        start_time = time.time()
        
        prompt = f"""
        As a senior investment analyst, create a comprehensive market summary based on the following data:

        Market Data:
        {json.dumps(market_data, indent=2)}

        Please provide:
        1. Executive summary of current market conditions
        2. Key market drivers and catalysts
        3. Sector performance analysis
        4. Technical outlook
        5. Investment opportunities and risks
        6. Short-term and medium-term outlook
        7. Action items for portfolio management

        Make it concise but comprehensive, suitable for investment committee review.
        """
        
        try:
            response = self._make_request_with_retry(prompt)
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.MARKET_SUMMARY,
                content=response,
                confidence=0.85,
                timestamp=datetime.now(),
                metadata={"market_data": market_data},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.MARKET_SUMMARY,
                content=f"Error generating market summary: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        current_time = time.time()
        self.rate_limiter._clean_old_requests(current_time)
        
        return {
            "requests_this_minute": len(self.rate_limiter.minute_requests),
            "requests_this_day": len(self.rate_limiter.day_requests),
            "minute_limit": self.rate_limiter.requests_per_minute,
            "day_limit": self.rate_limiter.requests_per_day,
            "current_backoff_time": self.rate_limiter.backoff_time,
            "model_name": self.model_name
        }

# Convenience functions
def create_gemini_analyzer(model_name: str = "gemini-1.5-flash") -> GeminiAnalyzer:
    """Create a Gemini analyzer instance"""
    return GeminiAnalyzer(model_name=model_name)

def quick_analysis(prompt: str, analysis_type: AnalysisType = AnalysisType.MARKET_SUMMARY) -> AnalysisResult:
    """Quick analysis with a custom prompt"""
    analyzer = create_gemini_analyzer()
    start_time = time.time()
    
    try:
        response = analyzer._make_request_with_retry(prompt)
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=analysis_type,
            content=response,
            confidence=0.70,
            timestamp=datetime.now(),
            metadata={"prompt": prompt},
            processing_time=processing_time
        )
        
    except Exception as e:
        return AnalysisResult(
            analysis_type=analysis_type,
            content=f"Error in analysis: {str(e)}",
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={"error": str(e)},
            processing_time=time.time() - start_time
        ) 