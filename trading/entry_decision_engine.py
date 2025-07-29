# Entry decision engine for Investment Committee
# Orchestrates the complete trade decision and execution pipeline

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trading.strategy.trade_filter import is_trade_eligible
from models.model_predictor import ModelPredictor
from models.neural_predictor import NeuralPredictor
from models.meta_model import MetaModel, ModelInput, TradeDecision, TradeSignal
from models.llm_analyzer import GeminiAnalyzer, create_gemini_analyzer, AnalysisType, AnalysisResult
from trading.execution.alpaca_client import AlpacaClient
from utils.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """Enhanced LLM analysis result for trade pipeline."""
    macro_analysis: Optional[AnalysisResult] = None
    sentiment_analysis: Optional[AnalysisResult] = None
    volatility_analysis: Optional[AnalysisResult] = None
    trade_summary: Optional[AnalysisResult] = None
    overall_recommendation: str = "NEUTRAL"
    risk_score: float = 0.5
    confidence_score: float = 0.0
    analysis_time: float = 0.0
    error_messages: List[str] = field(default_factory=list)


@dataclass
class TradeCandidate:
    """Trade candidate with all required data."""
    symbol: str
    historical_data: Dict[str, Any]
    technicals: Dict[str, float]
    options_data: Dict[str, Any]
    filter_passed: bool
    filter_metadata: Dict[str, Any]
    llm_analysis: Optional[LLMAnalysisResult] = None


@dataclass
class TradeExecution:
    """Trade execution record."""
    trade_id: str
    symbol: str
    signal: str
    confidence: float
    reasoning: List[str]
    execution_time: str
    execution_status: str
    order_details: Dict[str, Any]
    meta_decision: Dict[str, Any]
    llm_analysis: Optional[Dict[str, Any]] = None


class EntryDecisionEngine:
    """
    Entry decision engine that orchestrates the complete trading pipeline.
    
    Pipeline:
    1. Receive trade candidates from screening
    2. Run LLM analysis for macro/sentiment/volatility insights
    3. Run predictions through all ML models
    4. Combine predictions via meta-model (including LLM insights)
    5. Execute trades via Alpaca API
    6. Log all decisions and executions with LLM analysis
    """
    
    def __init__(self, paper_trading: bool = True, max_positions: int = 10, 
                 enable_llm_analysis: bool = True):
        """
        Initialize the entry decision engine.
        
        Args:
            paper_trading (bool): Whether to use paper trading mode
            max_positions (int): Maximum number of open positions
            enable_llm_analysis (bool): Whether to enable LLM analysis
        """
        self.paper_trading = paper_trading
        self.max_positions = max_positions
        self.enable_llm_analysis = enable_llm_analysis
        
        # Initialize components
        self.model_predictor = ModelPredictor()
        self.neural_mlp = NeuralPredictor(model_type='mlp')
        self.neural_lstm = NeuralPredictor(model_type='lstm')
        self.meta_model = MetaModel()
        
        # Initialize LLM analyzer
        if enable_llm_analysis:
            try:
                self.llm_analyzer = create_gemini_analyzer()
                self.llm_available = True
                logger.info("LLM analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"LLM analyzer unavailable: {e}")
                self.llm_available = False
        else:
            self.llm_available = False
            logger.info("LLM analysis disabled")
        
        # Initialize Alpaca client (with error handling)
        try:
            self.alpaca_client = AlpacaClient()
            self.alpaca_available = True
        except Exception as e:
            logger.warning(f"Alpaca client unavailable: {e}")
            self.alpaca_available = False
        
        # Initialize trade logger
        self.trade_logger = TradeLogger()
        
        # Track positions and executions
        self.active_positions = {}
        self.execution_log = []
        
        logger.info(f"Entry decision engine initialized (paper_trading={paper_trading}, llm_enabled={enable_llm_analysis})")
    
    def process_trade_candidates(self, candidates: List[TradeCandidate]) -> List[TradeExecution]:
        """
        Process multiple trade candidates through the complete pipeline.
        
        Args:
            candidates (List[TradeCandidate]): List of trade candidates
            
        Returns:
            List[TradeExecution]: List of trade executions
        """
        executions = []
        
        for candidate in candidates:
            try:
                execution = self.process_single_candidate(candidate)
                if execution:
                    executions.append(execution)
                    # Log trade execution with LLM analysis
                    self.trade_logger.log_trade(execution)
            except Exception as e:
                logger.error(f"Error processing candidate {candidate.symbol}: {e}")
                # Create error execution record
                error_execution = TradeExecution(
                    trade_id=f"ERROR_{candidate.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=candidate.symbol,
                    signal="ERROR",
                    confidence=0.0,
                    reasoning=[f"Processing error: {str(e)}"],
                    execution_time=datetime.now().isoformat(),
                    execution_status="ERROR",
                    order_details={},
                    meta_decision={'error': str(e)},
                    llm_analysis={'error': str(e)}
                )
                executions.append(error_execution)
                self.trade_logger.log_trade(error_execution)
        
        return executions
    
    def process_single_candidate(self, candidate: TradeCandidate) -> Optional[TradeExecution]:
        """
        Process a single trade candidate through the pipeline.
        
        Args:
            candidate (TradeCandidate): Trade candidate to process
            
        Returns:
            Optional[TradeExecution]: Trade execution if signal is BUY, None otherwise
        """
        logger.info(f"Processing candidate: {candidate.symbol}")
        
        # Step 1: Check if trade filter passed
        if not candidate.filter_passed:
            logger.info(f"Skipping {candidate.symbol} - failed trade filter")
            return None
        
        # Step 2: Check position limits
        if len(self.active_positions) >= self.max_positions:
            logger.info(f"Skipping {candidate.symbol} - maximum positions reached")
            return None
        
        # Step 3: Check if already have position in this symbol
        if candidate.symbol in self.active_positions:
            logger.info(f"Skipping {candidate.symbol} - already have position")
            return None
        
        # Step 4: Generate comprehensive LLM analysis
        if self.llm_available and self.enable_llm_analysis:
            candidate.llm_analysis = self._generate_comprehensive_llm_analysis(candidate)
            logger.info(f"LLM analysis completed for {candidate.symbol}")
        
        # Step 5: Generate predictions from all models
        model_inputs = self._generate_model_predictions(candidate)
        
        # Step 6: Get meta-model decision
        meta_decision = self.meta_model.predict_trade_signal(model_inputs)
        
        # Step 7: Execute trade if signal is BUY
        if meta_decision.signal == TradeSignal.BUY:
            execution = self._execute_trade(candidate, meta_decision)
            if execution:
                # Track position
                self.active_positions[candidate.symbol] = {
                    'entry_time': datetime.now(),
                    'trade_id': execution.trade_id,
                    'execution': execution
                }
                
                # Log execution
                self.execution_log.append(execution)
                
                logger.info(f"Trade executed: {candidate.symbol} - {execution.trade_id}")
                return execution
        else:
            logger.info(f"Meta-model passed on {candidate.symbol}: {meta_decision.reasoning[0]}")
        
        return None
    
    def _generate_comprehensive_llm_analysis(self, candidate: TradeCandidate) -> LLMAnalysisResult:
        """
        Generate comprehensive LLM analysis for a trade candidate.
        
        Args:
            candidate (TradeCandidate): Trade candidate
            
        Returns:
            LLMAnalysisResult: Comprehensive LLM analysis
        """
        start_time = datetime.now()
        llm_result = LLMAnalysisResult()
        
        try:
            # Prepare data for LLM analysis
            economic_data = self._prepare_economic_data(candidate)
            news_data = self._prepare_news_data(candidate)
            volatility_data = self._prepare_volatility_data(candidate)
            trade_data = self._prepare_trade_data(candidate)
            
            # 1. Macro Economic Analysis
            if economic_data:
                try:
                    llm_result.macro_analysis = self.llm_analyzer.analyze_macro_conditions(economic_data)
                    logger.info(f"Macro analysis completed for {candidate.symbol}")
                except Exception as e:
                    logger.error(f"Macro analysis failed for {candidate.symbol}: {e}")
                    llm_result.error_messages.append(f"Macro analysis error: {str(e)}")
            
            # 2. Sentiment Analysis
            if news_data:
                try:
                    llm_result.sentiment_analysis = self.llm_analyzer.analyze_sentiment(news_data)
                    logger.info(f"Sentiment analysis completed for {candidate.symbol}")
                except Exception as e:
                    logger.error(f"Sentiment analysis failed for {candidate.symbol}: {e}")
                    llm_result.error_messages.append(f"Sentiment analysis error: {str(e)}")
            
            # 3. Volatility Analysis
            if volatility_data:
                try:
                    llm_result.volatility_analysis = self.llm_analyzer.analyze_volatility_risks(volatility_data)
                    logger.info(f"Volatility analysis completed for {candidate.symbol}")
                except Exception as e:
                    logger.error(f"Volatility analysis failed for {candidate.symbol}: {e}")
                    llm_result.error_messages.append(f"Volatility analysis error: {str(e)}")
            
            # 4. Trade Summary and Recommendation
            try:
                llm_result.trade_summary = self.llm_analyzer.generate_market_summary(trade_data)
                logger.info(f"Trade summary completed for {candidate.symbol}")
            except Exception as e:
                logger.error(f"Trade summary failed for {candidate.symbol}: {e}")
                llm_result.error_messages.append(f"Trade summary error: {str(e)}")
            
            # 5. Generate overall recommendation and risk score
            llm_result.overall_recommendation, llm_result.risk_score, llm_result.confidence_score = \
                self._synthesize_llm_analysis(llm_result)
            
            # Calculate analysis time
            llm_result.analysis_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"LLM analysis completed for {candidate.symbol}: {llm_result.overall_recommendation} "
                       f"(risk: {llm_result.risk_score:.2f}, confidence: {llm_result.confidence_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error in comprehensive LLM analysis for {candidate.symbol}: {e}")
            llm_result.error_messages.append(f"General analysis error: {str(e)}")
            llm_result.overall_recommendation = "ERROR"
            llm_result.risk_score = 1.0
            llm_result.confidence_score = 0.0
        
        return llm_result
    
    def _prepare_economic_data(self, candidate: TradeCandidate) -> Dict[str, Any]:
        """Prepare economic data for LLM macro analysis."""
        technicals = candidate.technicals
        
        return {
            "symbol": candidate.symbol,
            "market_conditions": {
                "vix_level": technicals.get('vix_level', 20),
                "market_trend": technicals.get('market_trend', 0),
                "sector_rotation": technicals.get('sector_rotation', 'neutral'),
                "economic_cycle": technicals.get('economic_cycle', 'expansion')
            },
            "interest_rates": {
                "federal_funds_rate": technicals.get('fed_funds_rate', 5.25),
                "yield_curve": technicals.get('yield_curve', 'normal'),
                "rate_expectations": technicals.get('rate_expectations', 'stable')
            },
            "indicators": {
                "gdp_growth": technicals.get('gdp_growth', 2.0),
                "inflation_rate": technicals.get('inflation_rate', 3.0),
                "employment_data": technicals.get('unemployment_rate', 4.0),
                "consumer_sentiment": technicals.get('consumer_sentiment', 50)
            },
            "trade_context": {
                "strategy": "bull_put_spread",
                "current_price": candidate.historical_data.get('current_price', 100),
                "implied_volatility": technicals.get('iv_rank', 50)
            }
        }
    
    def _prepare_news_data(self, candidate: TradeCandidate) -> List[Dict[str, Any]]:
        """Prepare news data for LLM sentiment analysis."""
        # In a real implementation, this would fetch actual news
        # For now, create synthetic news based on market conditions
        technicals = candidate.technicals
        
        news_items = []
        
        # Market sentiment based on indicators
        if technicals.get('market_trend', 0) > 0.2:
            news_items.append({
                "headline": f"Market rallies as {candidate.symbol} shows strong momentum",
                "source": "Market News",
                "sentiment": "positive",
                "timestamp": datetime.now().isoformat(),
                "relevance": "high"
            })
        elif technicals.get('market_trend', 0) < -0.2:
            news_items.append({
                "headline": f"Market concerns weigh on {candidate.symbol} outlook",
                "source": "Market News",
                "sentiment": "negative",
                "timestamp": datetime.now().isoformat(),
                "relevance": "high"
            })
        
        # VIX-based sentiment
        if technicals.get('vix_level', 20) > 25:
            news_items.append({
                "headline": "Volatility spikes as market uncertainty increases",
                "source": "Financial Times",
                "sentiment": "negative",
                "timestamp": datetime.now().isoformat(),
                "relevance": "medium"
            })
        elif technicals.get('vix_level', 20) < 15:
            news_items.append({
                "headline": "Low volatility environment supports bullish sentiment",
                "source": "Wall Street Journal",
                "sentiment": "positive",
                "timestamp": datetime.now().isoformat(),
                "relevance": "medium"
            })
        
        # Add generic market news
        news_items.append({
            "headline": f"Options activity increases in {candidate.symbol}",
            "source": "Options Monitor",
            "sentiment": "neutral",
            "timestamp": datetime.now().isoformat(),
            "relevance": "medium"
        })
        
        return news_items
    
    def _prepare_volatility_data(self, candidate: TradeCandidate) -> Dict[str, Any]:
        """Prepare volatility data for LLM volatility analysis."""
        technicals = candidate.technicals
        
        return {
            "symbol": candidate.symbol,
            "volatility_metrics": {
                "implied_volatility": technicals.get('iv_rank', 50),
                "historical_volatility": technicals.get('historical_vol', 20),
                "vix_level": technicals.get('vix_level', 20),
                "volatility_skew": technicals.get('vol_skew', 0),
                "term_structure": technicals.get('vol_term_structure', 'normal')
            },
            "risk_metrics": {
                "beta": technicals.get('beta', 1.0),
                "correlation_spy": technicals.get('spy_correlation', 0.7),
                "max_drawdown": technicals.get('max_drawdown', -0.1),
                "sharpe_ratio": technicals.get('sharpe_ratio', 1.0)
            },
            "options_data": {
                "iv_rank": technicals.get('iv_rank', 50),
                "iv_percentile": technicals.get('iv_percentile', 50),
                "put_call_ratio": technicals.get('put_call_ratio', 0.8),
                "options_volume": technicals.get('options_volume', 1000000)
            },
            "trade_context": {
                "strategy": "bull_put_spread",
                "position_size": "moderate",
                "risk_tolerance": "conservative"
            }
        }
    
    def _prepare_trade_data(self, candidate: TradeCandidate) -> Dict[str, Any]:
        """Prepare comprehensive trade data for LLM market summary."""
        return {
            "symbol": candidate.symbol,
            "strategy": "bull_put_spread",
            "market_data": {
                "current_price": candidate.historical_data.get('current_price', 100),
                "price_trend": candidate.technicals.get('market_trend', 0),
                "volume_profile": candidate.technicals.get('volume_trend', 0),
                "support_levels": candidate.technicals.get('support_levels', []),
                "resistance_levels": candidate.technicals.get('resistance_levels', [])
            },
            "technical_indicators": candidate.technicals,
            "options_metrics": {
                "iv_rank": candidate.technicals.get('iv_rank', 50),
                "open_interest": candidate.technicals.get('open_interest', 1000),
                "put_call_ratio": candidate.technicals.get('put_call_ratio', 0.8)
            },
            "risk_assessment": {
                "market_risk": candidate.technicals.get('market_risk', 0.5),
                "earnings_risk": candidate.technicals.get('earnings_risk', 0.3),
                "volatility_risk": candidate.technicals.get('volatility_risk', 0.4)
            },
            "filter_results": candidate.filter_metadata
        }
    
    def _synthesize_llm_analysis(self, llm_result: LLMAnalysisResult) -> Tuple[str, float, float]:
        """
        Synthesize LLM analysis results into overall recommendation, risk score, and confidence.
        
        Args:
            llm_result (LLMAnalysisResult): LLM analysis results
            
        Returns:
            Tuple[str, float, float]: (recommendation, risk_score, confidence_score)
        """
        recommendations = []
        risk_scores = []
        confidence_scores = []
        
        # Process macro analysis
        if llm_result.macro_analysis:
            macro_content = llm_result.macro_analysis.content.lower()
            if 'bullish' in macro_content or 'positive' in macro_content:
                recommendations.append('BULLISH')
                risk_scores.append(0.3)
            elif 'bearish' in macro_content or 'negative' in macro_content:
                recommendations.append('BEARISH')
                risk_scores.append(0.8)
            else:
                recommendations.append('NEUTRAL')
                risk_scores.append(0.5)
            confidence_scores.append(llm_result.macro_analysis.confidence)
        
        # Process sentiment analysis
        if llm_result.sentiment_analysis:
            sentiment_content = llm_result.sentiment_analysis.content.lower()
            if 'bullish' in sentiment_content or 'positive' in sentiment_content:
                recommendations.append('BULLISH')
                risk_scores.append(0.4)
            elif 'bearish' in sentiment_content or 'negative' in sentiment_content:
                recommendations.append('BEARISH')
                risk_scores.append(0.7)
            else:
                recommendations.append('NEUTRAL')
                risk_scores.append(0.5)
            confidence_scores.append(llm_result.sentiment_analysis.confidence)
        
        # Process volatility analysis
        if llm_result.volatility_analysis:
            vol_content = llm_result.volatility_analysis.content.lower()
            if 'low volatility' in vol_content or 'stable' in vol_content:
                recommendations.append('BULLISH')
                risk_scores.append(0.3)
            elif 'high volatility' in vol_content or 'unstable' in vol_content:
                recommendations.append('BEARISH')
                risk_scores.append(0.8)
            else:
                recommendations.append('NEUTRAL')
                risk_scores.append(0.5)
            confidence_scores.append(llm_result.volatility_analysis.confidence)
        
        # Calculate overall recommendation
        if not recommendations:
            return "NEUTRAL", 0.5, 0.0
        
        bullish_count = recommendations.count('BULLISH')
        bearish_count = recommendations.count('BEARISH')
        neutral_count = recommendations.count('NEUTRAL')
        
        if bullish_count > bearish_count:
            overall_recommendation = 'BULLISH'
        elif bearish_count > bullish_count:
            overall_recommendation = 'BEARISH'
        else:
            overall_recommendation = 'NEUTRAL'
        
        # Calculate weighted risk score and confidence
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return overall_recommendation, avg_risk_score, avg_confidence
    
    def _generate_model_predictions(self, candidate: TradeCandidate) -> List[ModelInput]:
        """
        Generate predictions from all models.
        
        Args:
            candidate (TradeCandidate): Trade candidate
            
        Returns:
            List[ModelInput]: Model predictions
        """
        model_inputs = []
        
        # Enhanced model predictor (XGBoost)
        try:
            direction, confidence, metadata = self.model_predictor.predict_trade_signal(
                candidate.symbol, candidate.historical_data, candidate.technicals
            )
            model_inputs.append(ModelInput(
                model_name='xgboost',
                direction=direction,
                confidence=confidence,
                metadata=metadata
            ))
        except Exception as e:
            logger.error(f"Error in model predictor for {candidate.symbol}: {e}")
        
        # Neural MLP predictor
        try:
            neural_features = self._prepare_neural_features(candidate)
            direction, confidence, metadata = self.neural_mlp.predict_nn_signal(neural_features)
            model_inputs.append(ModelInput(
                model_name='neural_mlp',
                direction=direction,
                confidence=confidence,
                metadata=metadata
            ))
        except Exception as e:
            logger.error(f"Error in neural MLP for {candidate.symbol}: {e}")
        
        # Neural LSTM predictor
        try:
            neural_features = self._prepare_neural_features(candidate)
            direction, confidence, metadata = self.neural_lstm.predict_nn_signal(neural_features)
            model_inputs.append(ModelInput(
                model_name='neural_lstm',
                direction=direction,
                confidence=confidence,
                metadata=metadata
            ))
        except Exception as e:
            logger.error(f"Error in neural LSTM for {candidate.symbol}: {e}")
        
        # LLM analyzer input
        if candidate.llm_analysis:
            try:
                llm_input = self._convert_llm_to_model_input(candidate.llm_analysis)
                model_inputs.append(llm_input)
            except Exception as e:
                logger.error(f"Error converting LLM analysis for {candidate.symbol}: {e}")
        
        return model_inputs
    
    def _convert_llm_to_model_input(self, llm_analysis: LLMAnalysisResult) -> ModelInput:
        """
        Convert LLM analysis to ModelInput format.
        
        Args:
            llm_analysis (LLMAnalysisResult): LLM analysis result
            
        Returns:
            ModelInput: Model input for meta-model
        """
        # Map LLM recommendation to direction
        direction_map = {
            'BULLISH': 'BUY',
            'BEARISH': 'SELL',
            'NEUTRAL': 'HOLD'
        }
        
        direction = direction_map.get(llm_analysis.overall_recommendation, 'HOLD')
        
        # Convert risk score to confidence (inverse relationship)
        confidence = max(0.1, min(0.9, 1.0 - llm_analysis.risk_score))
        
        # Create metadata
        metadata = {
            'llm_recommendation': llm_analysis.overall_recommendation,
            'risk_score': llm_analysis.risk_score,
            'confidence_score': llm_analysis.confidence_score,
            'analysis_time': llm_analysis.analysis_time,
            'has_macro_analysis': llm_analysis.macro_analysis is not None,
            'has_sentiment_analysis': llm_analysis.sentiment_analysis is not None,
            'has_volatility_analysis': llm_analysis.volatility_analysis is not None,
            'has_trade_summary': llm_analysis.trade_summary is not None,
            'error_count': len(llm_analysis.error_messages)
        }
        
        return ModelInput(
            model_name='llm_gemini',
            direction=direction,
            confidence=confidence,
            metadata=metadata
        )
    
    def _prepare_neural_features(self, candidate: TradeCandidate) -> Dict[str, Any]:
        """
        Prepare features for neural network models.
        
        Args:
            candidate (TradeCandidate): Trade candidate
            
        Returns:
            Dict[str, Any]: Neural network features
        """
        # Create sequence from historical data
        sequence = []
        historical = candidate.historical_data
        
        if 'prices' in historical and 'volumes' in historical:
            prices = historical['prices']
            volumes = historical['volumes']
            highs = historical.get('highs', prices)
            lows = historical.get('lows', prices)
            
            for i in range(len(prices)):
                sequence.append([
                    prices[i] if i < len(prices) else prices[-1],
                    volumes[i] if i < len(volumes) else volumes[-1],
                    highs[i] if i < len(highs) else highs[-1],
                    lows[i] if i < len(lows) else lows[-1],
                    prices[i] if i < len(prices) else prices[-1]  # close = price
                ])
        
        return {
            'technicals': candidate.technicals,
            'sequence': sequence
        }
    
    def _execute_trade(self, candidate: TradeCandidate, meta_decision: TradeDecision) -> Optional[TradeExecution]:
        """
        Execute trade through Alpaca API.
        
        Args:
            candidate (TradeCandidate): Trade candidate
            meta_decision (TradeDecision): Meta-model decision
            
        Returns:
            Optional[TradeExecution]: Trade execution record
        """
        trade_id = f"BPS_{candidate.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Calculate position size and strike selection
            position_details = self._calculate_position_size(candidate, meta_decision)
            
            # Execute bull put spread (placeholder for now)
            execution_result = self._execute_bull_put_spread(candidate, position_details)
            
            # Convert LLM analysis to dict for logging
            llm_analysis_dict = None
            if candidate.llm_analysis:
                llm_analysis_dict = {
                    'overall_recommendation': candidate.llm_analysis.overall_recommendation,
                    'risk_score': candidate.llm_analysis.risk_score,
                    'confidence_score': candidate.llm_analysis.confidence_score,
                    'analysis_time': candidate.llm_analysis.analysis_time,
                    'macro_analysis': candidate.llm_analysis.macro_analysis.content[:500] if candidate.llm_analysis.macro_analysis else None,
                    'sentiment_analysis': candidate.llm_analysis.sentiment_analysis.content[:500] if candidate.llm_analysis.sentiment_analysis else None,
                    'volatility_analysis': candidate.llm_analysis.volatility_analysis.content[:500] if candidate.llm_analysis.volatility_analysis else None,
                    'trade_summary': candidate.llm_analysis.trade_summary.content[:500] if candidate.llm_analysis.trade_summary else None,
                    'error_messages': candidate.llm_analysis.error_messages
                }
            
            # Create execution record
            execution = TradeExecution(
                trade_id=trade_id,
                symbol=candidate.symbol,
                signal=meta_decision.signal.value,
                confidence=meta_decision.confidence,
                reasoning=meta_decision.reasoning,
                execution_time=datetime.now().isoformat(),
                execution_status=execution_result.get('status', 'PENDING'),
                order_details=execution_result,
                meta_decision=asdict(meta_decision),
                llm_analysis=llm_analysis_dict
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"Error executing trade for {candidate.symbol}: {e}")
            return TradeExecution(
                trade_id=trade_id,
                symbol=candidate.symbol,
                signal="ERROR",
                confidence=0.0,
                reasoning=[f"Execution error: {str(e)}"],
                execution_time=datetime.now().isoformat(),
                execution_status="ERROR",
                order_details={'error': str(e)},
                meta_decision=asdict(meta_decision),
                llm_analysis={'error': str(e)}
            )
    
    def _calculate_position_size(self, candidate: TradeCandidate, meta_decision: TradeDecision) -> Dict[str, Any]:
        """
        Calculate position size and strike selection for bull put spread.
        
        Args:
            candidate (TradeCandidate): Trade candidate
            meta_decision (TradeDecision): Meta-model decision
            
        Returns:
            Dict[str, Any]: Position details
        """
        # Get account information
        if self.alpaca_available:
            try:
                account = self.alpaca_client.get_account_info()
                buying_power = account.get('buying_power', 10000)
            except:
                buying_power = 10000  # Default for paper trading
        else:
            buying_power = 10000
        
        # Adjust position size based on LLM risk assessment
        base_risk_percent = 0.02  # 2% base risk
        if candidate.llm_analysis:
            risk_adjustment = candidate.llm_analysis.risk_score
            adjusted_risk_percent = base_risk_percent * (2.0 - risk_adjustment)  # Lower risk = larger position
        else:
            adjusted_risk_percent = base_risk_percent
        
        max_risk = buying_power * adjusted_risk_percent
        
        # Get current price
        current_price = candidate.historical_data.get('current_price', 100)
        
        # Calculate strike prices (placeholder logic)
        # Short put: 5% below current price
        # Long put: 10% below current price
        short_strike = current_price * 0.95
        long_strike = current_price * 0.90
        
        # Estimate credit received (placeholder)
        credit_per_contract = (short_strike - long_strike) * 0.3
        
        # Calculate number of contracts
        contracts = min(int(max_risk / ((short_strike - long_strike) * 100)), 10)
        
        return {
            'contracts': contracts,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'credit_per_contract': credit_per_contract,
            'max_risk': max_risk,
            'current_price': current_price,
            'buying_power': buying_power,
            'risk_adjustment': candidate.llm_analysis.risk_score if candidate.llm_analysis else 0.5,
            'adjusted_risk_percent': adjusted_risk_percent
        }
    
    def _execute_bull_put_spread(self, candidate: TradeCandidate, position_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute bull put spread order (placeholder).
        
        Args:
            candidate (TradeCandidate): Trade candidate
            position_details (Dict[str, Any]): Position details
            
        Returns:
            Dict[str, Any]: Execution result
        """
        # Placeholder for bull put spread execution
        # In real implementation, this would:
        # 1. Find appropriate options contracts
        # 2. Submit sell order for short put
        # 3. Submit buy order for long put
        # 4. Monitor fills and handle partial fills
        
        logger.info(f"Executing bull put spread for {candidate.symbol}")
        logger.info(f"Position details: {position_details}")
        
        # Include LLM analysis in execution log
        llm_info = ""
        if candidate.llm_analysis:
            llm_info = f" (LLM: {candidate.llm_analysis.overall_recommendation}, Risk: {candidate.llm_analysis.risk_score:.2f})"
        
        logger.info(f"LLM Analysis{llm_info}")
        
        # Simulate execution
        execution_result = {
            'status': 'FILLED' if self.paper_trading else 'PENDING',
            'symbol': candidate.symbol,
            'strategy': 'bull_put_spread',
            'contracts': position_details['contracts'],
            'short_strike': position_details['short_strike'],
            'long_strike': position_details['long_strike'],
            'credit_received': position_details['credit_per_contract'] * position_details['contracts'],
            'max_loss': (position_details['short_strike'] - position_details['long_strike']) * position_details['contracts'] * 100,
            'execution_time': datetime.now().isoformat(),
            'paper_trading': self.paper_trading,
            'llm_risk_adjustment': position_details.get('risk_adjustment', 0.5),
            'adjusted_risk_percent': position_details.get('adjusted_risk_percent', 0.02),
            'orders': []
        }
        
        # Add order details for tracking
        if self.paper_trading:
            execution_result['orders'] = [
                {
                    'side': 'SELL',
                    'instrument': f"{candidate.symbol}_PUT_{position_details['short_strike']}",
                    'quantity': position_details['contracts'],
                    'status': 'FILLED'
                },
                {
                    'side': 'BUY',
                    'instrument': f"{candidate.symbol}_PUT_{position_details['long_strike']}",
                    'quantity': position_details['contracts'],
                    'status': 'FILLED'
                }
            ]
        
        return execution_result
    
    def get_llm_usage_stats(self) -> Dict[str, Any]:
        """
        Get LLM usage statistics.
        
        Returns:
            Dict[str, Any]: LLM usage statistics
        """
        if self.llm_available:
            return self.llm_analyzer.get_usage_stats()
        else:
            return {'llm_available': False}
    
    def get_active_positions(self) -> Dict[str, Any]:
        """
        Get current active positions.
        
        Returns:
            Dict[str, Any]: Active positions
        """
        return self.active_positions
    
    def get_execution_log(self) -> List[TradeExecution]:
        """
        Get execution log.
        
        Returns:
            List[TradeExecution]: Execution log
        """
        return self.execution_log
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        executions = self.execution_log
        
        total_trades = len(executions)
        successful_trades = len([e for e in executions if e.execution_status == 'FILLED'])
        
        avg_confidence = sum(e.confidence for e in executions) / total_trades if total_trades > 0 else 0
        
        # LLM analysis statistics
        llm_analyses = [e for e in executions if e.llm_analysis and 'error' not in e.llm_analysis]
        llm_success_rate = len(llm_analyses) / total_trades if total_trades > 0 else 0
        
        avg_risk_score = 0
        if llm_analyses:
            avg_risk_score = sum(e.llm_analysis.get('risk_score', 0.5) for e in llm_analyses) / len(llm_analyses)
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'average_confidence': avg_confidence,
            'active_positions': len(self.active_positions),
            'last_trade_time': executions[-1].execution_time if executions else None,
            'llm_enabled': self.llm_available,
            'llm_success_rate': llm_success_rate,
            'average_risk_score': avg_risk_score,
            'llm_usage_stats': self.get_llm_usage_stats()
        }


# Helper functions
def create_trade_candidate(symbol: str, historical_data: Dict[str, Any], 
                         technicals: Dict[str, float]) -> TradeCandidate:
    """
    Create a trade candidate from basic inputs.
    
    Args:
        symbol (str): Stock symbol
        historical_data (Dict[str, Any]): Historical price data
        technicals (Dict[str, float]): Technical indicators
        
    Returns:
        TradeCandidate: Trade candidate object
    """
    # Create ticker data for trade filter
    ticker_data = {
        'ticker': symbol,
        'market_data': {
            'vix': technicals.get('vix_level', 20),
            'vvix': 85,  # Placeholder
            'spy_trend': 'up' if technicals.get('market_trend', 0) > 0 else 'down'
        },
        'ticker_data': {
            'avg_daily_volume': sum(historical_data.get('volumes', [1000000])[:20]) / 20,
            'iv_rank': technicals.get('volatility_rank', 50),
            'options_chain': {
                'put_leg_1': {'open_interest': 1000, 'bid_ask_spread': 0.05},
                'put_leg_2': {'open_interest': 800, 'bid_ask_spread': 0.08}
            }
        },
        'earnings': {'next_earnings_date': None}
    }
    
    # Check trade filter
    filter_passed = is_trade_eligible(ticker_data)
    
    return TradeCandidate(
        symbol=symbol,
        historical_data=historical_data,
        technicals=technicals,
        options_data={},
        filter_passed=filter_passed,
        filter_metadata=ticker_data
    )


def test_entry_decision_engine():
    """Test the entry decision engine with sample data."""
    print("Testing Entry Decision Engine...")
    
    # Create sample trade candidates
    candidates = []
    
    # Bullish candidate
    bullish_historical = {
        'prices': [150 + i * 0.5 for i in range(20)],
        'volumes': [50000000 + i * 1000000 for i in range(20)],
        'current_price': 160.0
    }
    bullish_technicals = {
        'rsi': 30,
        'vix_level': 16,
        'volatility_rank': 45,
        'market_trend': 0.6,
        'price_momentum': 0.4
    }
    candidates.append(create_trade_candidate('AAPL', bullish_historical, bullish_technicals))
    
    # Bearish candidate
    bearish_technicals = {
        'rsi': 75,
        'vix_level': 32,
        'volatility_rank': 85,
        'market_trend': -0.4,
        'price_momentum': -0.3
    }
    candidates.append(create_trade_candidate('TSLA', bullish_historical, bearish_technicals))
    
    # Initialize engine
    engine = EntryDecisionEngine(paper_trading=True)
    
    # Process candidates
    executions = engine.process_trade_candidates(candidates)
    
    print(f"\nProcessed {len(candidates)} candidates")
    print(f"Generated {len(executions)} executions")
    
    for execution in executions:
        print(f"\nExecution: {execution.symbol}")
        print(f"  Signal: {execution.signal}")
        print(f"  Confidence: {execution.confidence:.3f}")
        print(f"  Status: {execution.execution_status}")
        print(f"  Reasoning: {execution.reasoning[0] if execution.reasoning else 'N/A'}")
    
    # Show performance
    performance = engine.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Total Trades: {performance['total_trades']}")
    print(f"  Success Rate: {performance['success_rate']:.1%}")
    print(f"  Average Confidence: {performance['average_confidence']:.3f}")
    print(f"  Active Positions: {performance['active_positions']}")


if __name__ == "__main__":
    test_entry_decision_engine() 