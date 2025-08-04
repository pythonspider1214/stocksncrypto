import requests
import logging
from typing import Dict, List, Any
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available, using mock sentiment analysis")

from datetime import datetime, timedelta
from api_manager import api_manager

class SentimentAnalyzer:
    """News sentiment analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get news sentiment for symbol"""
        try:
            # Mock sentiment data for demonstration
            mock_sentiments = {
                'AAPL': {'sentiment': 'Positive', 'score': 0.15},
                'MSFT': {'sentiment': 'Positive', 'score': 0.12},
                'GOOGL': {'sentiment': 'Neutral', 'score': 0.02},
                'TSLA': {'sentiment': 'Negative', 'score': -0.08},
                'NVDA': {'sentiment': 'Positive', 'score': 0.18},
                'AMZN': {'sentiment': 'Neutral', 'score': 0.05},
                'META': {'sentiment': 'Negative', 'score': -0.03}
            }
            
            result = mock_sentiments.get(symbol, {'sentiment': 'Neutral', 'score': 0.0})
            
            return {
                'symbol': symbol,
                'overall_sentiment': result['sentiment'],
                'sentiment_score': result['score'],
                'article_count': 5,  # Mock data
                'confidence': 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {'error': str(e)}
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment"""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                return {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            else:
                # Mock sentiment analysis
                return {'polarity': 0.0, 'subjectivity': 0.5}
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return {'polarity': 0, 'subjectivity': 0}
