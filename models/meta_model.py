# Meta model module for Investment Committee
# Combines outputs from XGBoost, NN, and LLM to make final trade eligibility decisions

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradeSignal(Enum):
    """Trade signal enumeration."""
    BUY = "BUY"
    PASS = "PASS"


@dataclass
class ModelInput:
    """Input from individual models."""
    model_name: str
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class TradeDecision:
    """Final trade decision from meta-model."""
    signal: TradeSignal
    confidence: float
    reasoning: List[str]
    context: Dict[str, Any]
    model_inputs: List[ModelInput]


class MetaModel:
    """
    Meta-model that combines predictions from multiple models.
    Uses ensemble methods to make final trade decisions.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize meta-model.
        
        Args:
            weights (Dict[str, float], optional): Model weights for ensemble
        """
        # Default model weights
        self.weights = weights or {
            'xgboost': 0.4,      # Enhanced model predictor
            'neural_mlp': 0.25,   # Neural MLP
            'neural_lstm': 0.25,  # Neural LSTM
            'llm': 0.1           # LLM analyzer (placeholder)
        }
        
        # Validation thresholds
        self.min_confidence = 0.55  # Minimum confidence for BUY signal
        self.min_agreement = 0.6    # Minimum model agreement
        self.max_risk_factors = 3   # Maximum risk factors allowed
        
        logger.info(f"Meta-model initialized with weights: {self.weights}")
    
    def predict_trade_signal(self, model_inputs: List[ModelInput]) -> TradeDecision:
        """
        Generate final trade signal from multiple model inputs.
        
        Args:
            model_inputs (List[ModelInput]): Inputs from individual models
            
        Returns:
            TradeDecision: Final trade decision with reasoning
        """
        try:
            # Validate inputs
            if not model_inputs:
                return TradeDecision(
                    signal=TradeSignal.PASS,
                    confidence=0.0,
                    reasoning=["No model inputs provided"],
                    context={'error': 'No inputs'},
                    model_inputs=[]
                )
            
            # Extract predictions
            predictions = self._extract_predictions(model_inputs)
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(predictions)
            
            # Apply decision rules
            decision = self._apply_decision_rules(ensemble_metrics, model_inputs)
            
            logger.info(f"Meta-model decision: {decision.signal.value} with {decision.confidence:.3f} confidence")
            return decision
            
        except Exception as e:
            logger.error(f"Error in meta-model prediction: {e}")
            return TradeDecision(
                signal=TradeSignal.PASS,
                confidence=0.0,
                reasoning=[f"Error in meta-model: {str(e)}"],
                context={'error': str(e)},
                model_inputs=model_inputs
            )
    
    def _extract_predictions(self, model_inputs: List[ModelInput]) -> Dict[str, Dict[str, Any]]:
        """
        Extract and normalize predictions from model inputs.
        
        Args:
            model_inputs (List[ModelInput]): Raw model inputs
            
        Returns:
            Dict[str, Dict[str, Any]]: Normalized predictions
        """
        predictions = {}
        
        for model_input in model_inputs:
            # Convert direction to numerical score
            direction_score = self._direction_to_score(model_input.direction)
            
            predictions[model_input.model_name] = {
                'direction': model_input.direction,
                'confidence': model_input.confidence,
                'score': direction_score,
                'weight': self.weights.get(model_input.model_name, 0.1),
                'metadata': model_input.metadata
            }
        
        return predictions
    
    def _direction_to_score(self, direction: str) -> float:
        """
        Convert direction string to numerical score.
        
        Args:
            direction (str): Direction string
            
        Returns:
            float: Numerical score (-1 to 1)
        """
        direction_map = {
            'BULLISH': 1.0,
            'NEUTRAL': 0.0,
            'BEARISH': -1.0
        }
        return direction_map.get(direction.upper(), 0.0)
    
    def _calculate_ensemble_metrics(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate ensemble metrics from individual predictions.
        
        Args:
            predictions (Dict[str, Dict[str, Any]]): Individual model predictions
            
        Returns:
            Dict[str, Any]: Ensemble metrics
        """
        if not predictions:
            return {'weighted_score': 0.0, 'weighted_confidence': 0.0, 'agreement': 0.0}
        
        # Calculate weighted score and confidence
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        directions = []
        confidences = []
        
        for model_name, pred in predictions.items():
            weight = pred['weight']
            weighted_score += pred['score'] * pred['confidence'] * weight
            weighted_confidence += pred['confidence'] * weight
            total_weight += weight
            
            directions.append(pred['direction'])
            confidences.append(pred['confidence'])
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_score /= total_weight
            weighted_confidence /= total_weight
        
        # Calculate agreement (consensus)
        agreement = self._calculate_agreement(directions)
        
        # Calculate risk assessment
        risk_factors = self._assess_risk_factors(predictions)
        
        return {
            'weighted_score': weighted_score,
            'weighted_confidence': weighted_confidence,
            'agreement': agreement,
            'risk_factors': risk_factors,
            'model_count': len(predictions),
            'directions': directions,
            'confidences': confidences
        }
    
    def _calculate_agreement(self, directions: List[str]) -> float:
        """
        Calculate agreement score between model directions.
        
        Args:
            directions (List[str]): Model directions
            
        Returns:
            float: Agreement score (0.0 to 1.0)
        """
        if not directions:
            return 0.0
        
        # Count direction occurrences
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Calculate maximum consensus
        max_count = max(direction_counts.values())
        agreement = max_count / len(directions)
        
        return agreement
    
    def _assess_risk_factors(self, predictions: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Assess risk factors from model predictions.
        
        Args:
            predictions (Dict[str, Dict[str, Any]]): Model predictions
            
        Returns:
            List[str]: Risk factors identified
        """
        risk_factors = []
        
        # Check for low confidence predictions
        low_confidence_models = [
            name for name, pred in predictions.items() 
            if pred['confidence'] < 0.5
        ]
        if low_confidence_models:
            risk_factors.append(f"Low confidence from: {', '.join(low_confidence_models)}")
        
        # Check for bearish signals
        bearish_models = [
            name for name, pred in predictions.items() 
            if pred['direction'] == 'BEARISH'
        ]
        if bearish_models:
            risk_factors.append(f"Bearish signals from: {', '.join(bearish_models)}")
        
        # Check metadata for specific risk factors
        for model_name, pred in predictions.items():
            metadata = pred['metadata']
            if isinstance(metadata, dict):
                # Check for risk factors in metadata
                model_risk_factors = metadata.get('risk_factors', [])
                if model_risk_factors:
                    risk_factors.extend([f"{model_name}: {rf}" for rf in model_risk_factors[:2]])
        
        return risk_factors
    
    def _apply_decision_rules(self, ensemble_metrics: Dict[str, Any], 
                           model_inputs: List[ModelInput]) -> TradeDecision:
        """
        Apply decision rules to determine final trade signal.
        
        Args:
            ensemble_metrics (Dict[str, Any]): Ensemble metrics
            model_inputs (List[ModelInput]): Original model inputs
            
        Returns:
            TradeDecision: Final trade decision
        """
        reasoning = []
        context = ensemble_metrics.copy()
        
        # Extract metrics
        weighted_score = ensemble_metrics['weighted_score']
        weighted_confidence = ensemble_metrics['weighted_confidence']
        agreement = ensemble_metrics['agreement']
        risk_factors = ensemble_metrics['risk_factors']
        
        # Decision logic
        signal = TradeSignal.PASS  # Default to PASS
        final_confidence = weighted_confidence
        
        # Rule 1: Minimum confidence threshold
        if weighted_confidence < self.min_confidence:
            reasoning.append(f"Confidence too low: {weighted_confidence:.3f} < {self.min_confidence}")
            signal = TradeSignal.PASS
        
        # Rule 2: Minimum agreement threshold
        elif agreement < self.min_agreement:
            reasoning.append(f"Low model agreement: {agreement:.3f} < {self.min_agreement}")
            signal = TradeSignal.PASS
        
        # Rule 3: Too many risk factors
        elif len(risk_factors) > self.max_risk_factors:
            reasoning.append(f"Too many risk factors: {len(risk_factors)} > {self.max_risk_factors}")
            signal = TradeSignal.PASS
        
        # Rule 4: Bearish or neutral overall signal
        elif weighted_score <= 0.1:
            reasoning.append(f"Weak/bearish signal: score {weighted_score:.3f}")
            signal = TradeSignal.PASS
        
        # Rule 5: All conditions met for BUY
        else:
            reasoning.append(f"Strong bullish consensus: score {weighted_score:.3f}")
            reasoning.append(f"High confidence: {weighted_confidence:.3f}")
            reasoning.append(f"Good model agreement: {agreement:.3f}")
            signal = TradeSignal.BUY
        
        # Add supporting context
        if signal == TradeSignal.BUY:
            reasoning.append(f"Risk factors: {len(risk_factors)}")
            reasoning.append(f"Model consensus: {ensemble_metrics['model_count']} models")
        
        # Adjust confidence based on decision
        if signal == TradeSignal.PASS:
            final_confidence = min(final_confidence, 0.4)  # Cap confidence for PASS signals
        
        return TradeDecision(
            signal=signal,
            confidence=final_confidence,
            reasoning=reasoning,
            context=context,
            model_inputs=model_inputs
        )
    
    def add_llm_input(self, llm_analysis: Dict[str, Any]) -> ModelInput:
        """
        Create model input from LLM analysis (placeholder).
        
        Args:
            llm_analysis (Dict[str, Any]): LLM analysis results
            
        Returns:
            ModelInput: Formatted model input
        """
        # Placeholder implementation
        # In real implementation, this would process LLM output
        
        # Extract or default values
        sentiment = llm_analysis.get('sentiment', 'neutral')
        macro_risk = llm_analysis.get('macro_risk', 0.5)
        news_sentiment = llm_analysis.get('news_sentiment', 0.5)
        
        # Convert sentiment to direction
        if sentiment == 'bullish' or news_sentiment > 0.6:
            direction = 'BULLISH'
        elif sentiment == 'bearish' or news_sentiment < 0.4:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Calculate confidence based on macro risk
        confidence = max(0.3, min(0.8, 1.0 - macro_risk))
        
        return ModelInput(
            model_name='llm',
            direction=direction,
            confidence=confidence,
            metadata={
                'sentiment': sentiment,
                'macro_risk': macro_risk,
                'news_sentiment': news_sentiment,
                'analysis_time': datetime.now().isoformat(),
                'placeholder': True
            }
        )
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update model weights.
        
        Args:
            new_weights (Dict[str, float]): New model weights
        """
        self.weights.update(new_weights)
        logger.info(f"Updated model weights: {self.weights}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get meta-model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'weights': self.weights,
            'thresholds': {
                'min_confidence': self.min_confidence,
                'min_agreement': self.min_agreement,
                'max_risk_factors': self.max_risk_factors
            },
            'version': '1.0.0',
            'last_updated': datetime.now().isoformat()
        }


# Convenience functions
def create_model_input(model_name: str, direction: str, confidence: float, 
                      metadata: Optional[Dict[str, Any]] = None) -> ModelInput:
    """
    Create a ModelInput object.
    
    Args:
        model_name (str): Name of the model
        direction (str): Direction prediction
        confidence (float): Confidence score
        metadata (Dict[str, Any], optional): Additional metadata
        
    Returns:
        ModelInput: Formatted model input
    """
    return ModelInput(
        model_name=model_name,
        direction=direction,
        confidence=confidence,
        metadata=metadata or {}
    )


def combine_predictions(xgboost_pred: Tuple[str, float, Dict[str, Any]],
                       neural_pred: Tuple[str, float, Dict[str, Any]],
                       llm_analysis: Optional[Dict[str, Any]] = None) -> TradeDecision:
    """
    Convenience function to combine predictions from multiple models.
    
    Args:
        xgboost_pred (Tuple[str, float, Dict[str, Any]]): XGBoost prediction
        neural_pred (Tuple[str, float, Dict[str, Any]]): Neural network prediction
        llm_analysis (Dict[str, Any], optional): LLM analysis
        
    Returns:
        TradeDecision: Final trade decision
    """
    meta_model = MetaModel()
    
    # Create model inputs
    model_inputs = [
        create_model_input('xgboost', xgboost_pred[0], xgboost_pred[1], xgboost_pred[2]),
        create_model_input('neural_mlp', neural_pred[0], neural_pred[1], neural_pred[2])
    ]
    
    # Add LLM input if provided
    if llm_analysis:
        llm_input = meta_model.add_llm_input(llm_analysis)
        model_inputs.append(llm_input)
    
    return meta_model.predict_trade_signal(model_inputs)


def test_meta_model():
    """Test the meta-model with sample predictions."""
    print("Testing Meta-Model...")
    
    # Create sample predictions
    xgboost_pred = ('BULLISH', 0.75, {'feature_importance': {'rsi': 0.3}})
    neural_pred = ('BULLISH', 0.65, {'probabilities': {'bullish': 0.65, 'neutral': 0.25, 'bearish': 0.10}})
    llm_analysis = {'sentiment': 'bullish', 'macro_risk': 0.3, 'news_sentiment': 0.7}
    
    # Test combination
    decision = combine_predictions(xgboost_pred, neural_pred, llm_analysis)
    
    print(f"\nDecision: {decision.signal.value}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Risk factors: {len(decision.context.get('risk_factors', []))}")
    
    # Test different scenarios
    print("\nTesting different scenarios:")
    
    # Low confidence scenario
    low_conf_pred = ('BULLISH', 0.35, {})
    decision = combine_predictions(low_conf_pred, neural_pred)
    print(f"Low confidence: {decision.signal.value} - {decision.reasoning[0]}")
    
    # Bearish scenario
    bearish_pred = ('BEARISH', 0.80, {})
    decision = combine_predictions(bearish_pred, neural_pred)
    print(f"Bearish signal: {decision.signal.value} - {decision.reasoning[0]}")
    
    # Mixed signals
    mixed_pred = ('BEARISH', 0.60, {})
    decision = combine_predictions(mixed_pred, neural_pred)
    print(f"Mixed signals: {decision.signal.value} - Agreement: {decision.context.get('agreement', 0):.3f}")


if __name__ == "__main__":
    test_meta_model() 