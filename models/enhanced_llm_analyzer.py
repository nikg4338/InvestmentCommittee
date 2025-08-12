"""
Enhanced LLM analyzer module using Google Gemini API
Integrates with meta model and provides trading-specific analysis
"""

import asyncio
import time
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

try:
    from config.settings import get_gemini_config, validate_gemini_api_key
except ImportError:
    # Fallback to environment variables
    def get_gemini_config():
        return {"api_key": os.getenv("GEMINI_API_KEY")}
    
    def validate_gemini_api_key():
        return bool(os.getenv("GEMINI_API_KEY"))

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    MACRO_ANALYSIS = "macro_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_DECISION = "trade_decision"
    PORTFOLIO_REVIEW = "portfolio_review"
    META_MODEL_INTEGRATION = "meta_model_integration"
    MARKET_SUMMARY = "market_summary"


@dataclass
class AnalysisResult:
    """Enhanced result of LLM analysis with meta model integration"""
    analysis_type: AnalysisType
    symbol: Optional[str]
    timestamp: datetime
    summary: str
    confidence: float  # 0.0 to 1.0
    recommendation: str  # BUY, SELL, HOLD
    reasoning: str
    risk_factors: List[str]
    supporting_data: Dict[str, Any]
    meta_model_score: Optional[float] = None
    ml_model_predictions: Optional[Dict[str, float]] = None
    consensus_score: Optional[float] = None
    content: Optional[str] = None  # Full LLM response
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None


class MetaModelIntegrator:
    """Integrates LLM analysis with machine learning meta model"""
    
    def __init__(self, meta_model_path: str = "models/production/meta_model.pkl"):
        """Initialize meta model integrator"""
        self.meta_model_path = meta_model_path
        self.meta_model = None
        self.load_meta_model()
    
    def load_meta_model(self):
        """Load the meta model if it exists"""
        try:
            if os.path.exists(self.meta_model_path):
                self.meta_model = joblib.load(self.meta_model_path)
                logger.info(f"✅ Loaded meta model from {self.meta_model_path}")
            else:
                logger.warning(f"Meta model not found at {self.meta_model_path}")
        except Exception as e:
            logger.error(f"Failed to load meta model: {e}")
            self.meta_model = None
    
    def create_meta_features(self, 
                           ml_predictions: Dict[str, float],
                           llm_analysis: AnalysisResult,
                           market_context: Dict[str, Any]) -> np.ndarray:
        """Create feature vector for meta model"""
        features = []
        
        # ML model predictions
        model_preds = []
        for model_name in ['optimized_catboost', 'random_forest', 'svm']:
            model_preds.append(ml_predictions.get(model_name, 0.5))
        features.extend(model_preds)
        
        # LLM features
        features.append(llm_analysis.confidence)
        
        # Convert recommendation to numeric
        rec_map = {"SELL": 0.0, "HOLD": 0.5, "BUY": 1.0}
        features.append(rec_map.get(llm_analysis.recommendation, 0.5))
        
        # Risk score (inverse of confidence)
        features.append(1.0 - llm_analysis.confidence)
        
        # Market context features
        features.append(market_context.get('vix_level', 20.0) / 100.0)  # Normalize VIX
        features.append(market_context.get('spy_change', 0.0) / 100.0)  # Normalize SPY change
        
        # Sentiment score
        sentiment_map = {"BEARISH": 0.0, "NEUTRAL": 0.5, "BULLISH": 1.0}
        sentiment = market_context.get('market_sentiment', 'NEUTRAL')
        features.append(sentiment_map.get(sentiment, 0.5))
        
        return np.array(features).reshape(1, -1)
    
    def get_meta_prediction(self, 
                          ml_predictions: Dict[str, float],
                          llm_analysis: AnalysisResult,
                          market_context: Dict[str, Any]) -> Optional[float]:
        """Get prediction from meta model"""
        if not self.meta_model:
            return None
        
        try:
            features = self.create_meta_features(ml_predictions, llm_analysis, market_context)
            
            if hasattr(self.meta_model, 'predict_proba'):
                prediction = self.meta_model.predict_proba(features)[:, 1][0]
            else:
                prediction = self.meta_model.predict(features)[0]
                # Apply sigmoid if needed
                if prediction < 0 or prediction > 1:
                    prediction = 1 / (1 + np.exp(-prediction))
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Meta model prediction failed: {e}")
            return None
    
    def calculate_consensus_score(self,
                                ml_predictions: Dict[str, float],
                                llm_score: float,
                                meta_score: Optional[float] = None) -> float:
        """Calculate consensus score from all models"""
        scores = list(ml_predictions.values()) + [llm_score]
        
        if meta_score is not None:
            scores.append(meta_score)
        
        # Weighted average (can be enhanced)
        weights = [0.3, 0.3, 0.2, 0.2]  # Adjust based on model performance
        if len(scores) > len(weights):
            weights = weights + [0.1] * (len(scores) - len(weights))
        
        consensus = sum(score * weight for score, weight in zip(scores, weights[:len(scores)]))
        consensus = consensus / sum(weights[:len(scores)])
        
        return consensus


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
    """Enhanced Gemini API client for investment analysis with meta model integration"""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "gemini-2.0-flash-exp",
                 enable_meta_model: bool = True):
        """Initialize Gemini analyzer"""
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google AI packages not available. Please install with: pip install google-generativeai")
        
        # Set up API key
        if api_key:
            genai.configure(api_key=api_key)
        elif validate_gemini_api_key():
            config = get_gemini_config()
            genai.configure(api_key=config["api_key"])
        else:
            # Try environment variable
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY environment variable")
            genai.configure(api_key=api_key)
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.rate_limiter = RateLimiter()
        
        # Initialize meta model integrator
        self.meta_integrator = MetaModelIntegrator() if enable_meta_model else None
        
        # Safety settings for financial analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        logger.info(f"Initialized enhanced Gemini analyzer with model: {model_name}")
    
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
    
    def analyze_trade_opportunity(self, 
                                symbol: str,
                                market_data: Dict[str, Any],
                                analysis_type: AnalysisType = AnalysisType.TRADE_DECISION) -> AnalysisResult:
        """Analyze a specific trade opportunity with meta model integration"""
        start_time = time.time()
        
        # Extract relevant data
        ml_predictions = market_data.get('model_predictions', {})
        technical_data = market_data.get('technical_data', {})
        market_context = market_data.get('market_context', {})
        
        prompt = f"""
        As an expert quantitative analyst and portfolio manager, analyze this trading opportunity:

        SYMBOL: {symbol}
        
        MACHINE LEARNING MODEL PREDICTIONS:
        {json.dumps(ml_predictions, indent=2)}
        
        TECHNICAL INDICATORS:
        {json.dumps(technical_data, indent=2)}
        
        MARKET CONTEXT:
        {json.dumps(market_context, indent=2)}
        
        Please provide a comprehensive analysis including:
        
        1. RECOMMENDATION: Provide a clear BUY, SELL, or HOLD recommendation
        2. CONFIDENCE: Rate your confidence (0-100)
        3. REASONING: Explain your logic considering:
           - ML model signals and their reliability
           - Technical indicator alignment
           - Current market environment
           - Risk-reward profile
        4. RISK FACTORS: List 3-5 key risks to monitor
        5. ENTRY/EXIT STRATEGY: Suggest optimal timing and levels
        6. POSITION SIZING: Recommend position size relative to portfolio
        
        Format your response as structured JSON with these exact keys:
        {{
            "recommendation": "BUY/SELL/HOLD",
            "confidence": 85,
            "reasoning": "Detailed explanation...",
            "risk_factors": ["Risk 1", "Risk 2", "Risk 3"],
            "entry_strategy": "Entry timing and levels...",
            "position_size": "Position sizing recommendation...",
            "ml_model_assessment": "Assessment of ML predictions...",
            "technical_assessment": "Technical analysis summary...",
            "market_environment_impact": "How current market affects this trade..."
        }}
        """
        
        try:
            response = self._make_request_with_retry(prompt)
            processing_time = time.time() - start_time
            
            # Parse LLM response
            analysis_data = self._parse_llm_response(response)
            
            # Convert LLM confidence to 0-1 scale
            llm_confidence = analysis_data.get('confidence', 70) / 100.0
            
            # Get meta model prediction if available
            meta_score = None
            consensus_score = None
            
            if self.meta_integrator and ml_predictions:
                # Create temporary analysis result for meta model
                temp_analysis = AnalysisResult(
                    analysis_type=analysis_type,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    summary="",
                    confidence=llm_confidence,
                    recommendation=analysis_data.get('recommendation', 'HOLD'),
                    reasoning="",
                    risk_factors=[],
                    supporting_data={}
                )
                
                meta_score = self.meta_integrator.get_meta_prediction(
                    ml_predictions, temp_analysis, market_context
                )
                
                # Calculate consensus score
                consensus_score = self.meta_integrator.calculate_consensus_score(
                    ml_predictions, llm_confidence, meta_score
                )
            
            # Create final analysis result
            result = AnalysisResult(
                analysis_type=analysis_type,
                symbol=symbol,
                timestamp=datetime.now(),
                summary=analysis_data.get('reasoning', response[:200] + '...'),
                confidence=llm_confidence,
                recommendation=analysis_data.get('recommendation', 'HOLD'),
                reasoning=analysis_data.get('reasoning', ''),
                risk_factors=analysis_data.get('risk_factors', []),
                supporting_data={
                    'entry_strategy': analysis_data.get('entry_strategy', ''),
                    'position_size': analysis_data.get('position_size', ''),
                    'ml_model_assessment': analysis_data.get('ml_model_assessment', ''),
                    'technical_assessment': analysis_data.get('technical_assessment', ''),
                    'market_environment_impact': analysis_data.get('market_environment_impact', '')
                },
                meta_model_score=meta_score,
                ml_model_predictions=ml_predictions,
                consensus_score=consensus_score,
                content=response,
                processing_time=processing_time
            )
            
            logger.info(f"Trade analysis completed for {symbol} - {result.recommendation} ({result.confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in trade analysis for {symbol}: {e}")
            return AnalysisResult(
                analysis_type=analysis_type,
                symbol=symbol,
                timestamp=datetime.now(),
                summary=f"Error analyzing {symbol}: {str(e)}",
                confidence=0.0,
                recommendation="HOLD",
                reasoning=f"Analysis failed: {str(e)}",
                risk_factors=["Analysis failure"],
                supporting_data={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response, handling both JSON and text formats"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to text parsing
        analysis_data = {
            'recommendation': 'HOLD',
            'confidence': 50,
            'reasoning': response,
            'risk_factors': [],
            'entry_strategy': '',
            'position_size': '',
            'ml_model_assessment': '',
            'technical_assessment': '',
            'market_environment_impact': ''
        }
        
        # Extract recommendation
        response_upper = response.upper()
        if 'BUY' in response_upper and 'SELL' not in response_upper:
            analysis_data['recommendation'] = 'BUY'
        elif 'SELL' in response_upper:
            analysis_data['recommendation'] = 'SELL'
        
        # Extract confidence (look for numbers followed by %)
        import re
        confidence_match = re.search(r'(\d{1,3})%', response)
        if confidence_match:
            analysis_data['confidence'] = int(confidence_match.group(1))
        
        return analysis_data
    
    def analyze_portfolio_allocation(self, 
                                   portfolio_data: Dict[str, Any],
                                   signals: List[Dict[str, Any]]) -> AnalysisResult:
        """Analyze portfolio allocation and rebalancing needs"""
        start_time = time.time()
        
        prompt = f"""
        As a portfolio manager, analyze the current portfolio allocation and trading signals:

        CURRENT PORTFOLIO:
        {json.dumps(portfolio_data, indent=2)}
        
        TRADING SIGNALS:
        {json.dumps(signals[:10], indent=2)}  # Limit to avoid token limits
        
        Please provide:
        1. Portfolio health assessment
        2. Diversification analysis
        3. Risk concentration analysis
        4. Rebalancing recommendations
        5. New position recommendations based on signals
        6. Risk management suggestions
        
        Focus on practical portfolio management insights.
        """
        
        try:
            response = self._make_request_with_retry(prompt)
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.PORTFOLIO_REVIEW,
                symbol=None,
                timestamp=datetime.now(),
                summary=response[:200] + '...',
                confidence=0.80,
                recommendation="REBALANCE",
                reasoning=response,
                risk_factors=["Portfolio concentration", "Market timing"],
                supporting_data={"portfolio_data": portfolio_data, "signals_count": len(signals)},
                content=response,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.PORTFOLIO_REVIEW,
                symbol=None,
                timestamp=datetime.now(),
                summary=f"Error analyzing portfolio: {str(e)}",
                confidence=0.0,
                recommendation="HOLD",
                reasoning=f"Portfolio analysis failed: {str(e)}",
                risk_factors=["Analysis failure"],
                supporting_data={"error": str(e)},
                processing_time=time.time() - start_time
            )


# Convenience functions for backward compatibility
def create_gemini_analyzer(api_key: str = None, 
                         model_name: str = "gemini-2.0-flash-exp",
                         enable_meta_model: bool = True) -> GeminiAnalyzer:
    """Create a Gemini analyzer instance"""
    return GeminiAnalyzer(api_key=api_key, model_name=model_name, enable_meta_model=enable_meta_model)


# Export key classes and functions
__all__ = [
    'GeminiAnalyzer',
    'AnalysisType', 
    'AnalysisResult',
    'MetaModelIntegrator',
    'create_gemini_analyzer'
]
