"""
Clinical Decision Support Tools for Alzheimer's Disease Detection

This module provides advanced clinical features including:
- Risk scoring and stratification
- Progression tracking
- Clinical decision support
- Biomarker trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


class RiskScorer:
    """Calculate and stratify Alzheimer's disease risk scores."""
    
    def __init__(self):
        """Initialize risk scoring system."""
        # Risk factors and their weights
        self.risk_factors = {
            'age': {'weight': 0.3, 'high_threshold': 75, 'medium_threshold': 65},
            'mmse': {'weight': 0.4, 'high_threshold': 24, 'medium_threshold': 27},
            'nwbv': {'weight': 0.2, 'high_threshold': 0.70, 'medium_threshold': 0.75},
            'education': {'weight': 0.1, 'high_threshold': 12, 'medium_threshold': 16}
        }
    
    def calculate_risk_score(self, features: Dict) -> Dict:
        """
        Calculate comprehensive risk score.
        
        Args:
            features: Dictionary with patient features
            
        Returns:
            Dictionary with risk score and stratification
        """
        score = 0.0
        risk_details = {}
        
        # Age risk (higher age = higher risk)
        age = features.get('Age', 0)
        if age >= self.risk_factors['age']['high_threshold']:
            age_score = 1.0
        elif age >= self.risk_factors['age']['medium_threshold']:
            age_score = 0.6
        else:
            age_score = 0.3
        score += age_score * self.risk_factors['age']['weight']
        risk_details['age_risk'] = age_score
        
        # MMSE risk (lower MMSE = higher risk)
        mmse = features.get('MMSE', 30)
        if mmse <= self.risk_factors['mmse']['high_threshold']:
            mmse_score = 1.0
        elif mmse <= self.risk_factors['mmse']['medium_threshold']:
            mmse_score = 0.6
        else:
            mmse_score = 0.2
        score += mmse_score * self.risk_factors['mmse']['weight']
        risk_details['mmse_risk'] = mmse_score
        
        # Brain volume risk (lower nWBV = higher risk)
        nwbv = features.get('nWBV', 1.0)
        if nwbv <= self.risk_factors['nwbv']['high_threshold']:
            nwbv_score = 1.0
        elif nwbv <= self.risk_factors['nwbv']['medium_threshold']:
            nwbv_score = 0.6
        else:
            nwbv_score = 0.2
        score += nwbv_score * self.risk_factors['nwbv']['weight']
        risk_details['nwbv_risk'] = nwbv_score
        
        # Education risk (lower education = slightly higher risk)
        educ = features.get('EDUC', 16)
        if educ <= self.risk_factors['education']['high_threshold']:
            educ_score = 0.3
        elif educ <= self.risk_factors['education']['medium_threshold']:
            educ_score = 0.15
        else:
            educ_score = 0.05
        score += educ_score * self.risk_factors['education']['weight']
        risk_details['education_risk'] = educ_score
        
        # Normalize score to 0-100
        risk_score = min(100, score * 100)
        
        # Stratify risk
        if risk_score >= 70:
            risk_level = "High"
            risk_color = "red"
        elif risk_score >= 40:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_details': risk_details,
            'interpretation': self._get_interpretation(risk_level)
        }
    
    def _get_interpretation(self, risk_level: str) -> str:
        """Get clinical interpretation of risk level."""
        interpretations = {
            'High': 'High risk for dementia. Recommend comprehensive neuropsychological evaluation and follow-up imaging.',
            'Moderate': 'Moderate risk. Monitor closely and consider follow-up assessment in 6-12 months.',
            'Low': 'Low risk. Continue routine screening as per standard protocols.'
        }
        return interpretations.get(risk_level, 'Risk assessment completed.')


class ProgressionTracker:
    """Track disease progression over time."""
    
    def __init__(self):
        """Initialize progression tracker."""
        self.history_file = "data/patient_history.json"
    
    def add_visit(self, patient_id: str, visit_date: str, features: Dict, prediction: Dict):
        """
        Add a new visit to patient history.
        
        Args:
            patient_id: Unique patient identifier
            visit_date: Date of visit (YYYY-MM-DD)
            features: Patient features
            prediction: Model prediction results
        """
        history = self.load_history()
        
        if patient_id not in history:
            history[patient_id] = []
        
        visit = {
            'date': visit_date,
            'features': features,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        history[patient_id].append(visit)
        history[patient_id].sort(key=lambda x: x['date'])
        
        self.save_history(history)
    
    def get_progression(self, patient_id: str) -> Dict:
        """
        Analyze progression for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Progression analysis
        """
        history = self.load_history()
        
        if patient_id not in history or len(history[patient_id]) < 2:
            return {'error': 'Insufficient history for progression analysis'}
        
        visits = history[patient_id]
        
        # Calculate trends
        mmse_trend = self._calculate_trend([v['features'].get('MMSE', 0) for v in visits])
        nwbv_trend = self._calculate_trend([v['features'].get('nWBV', 0) for v in visits])
        risk_trend = self._calculate_trend(
            [v['prediction'].get('probability_demented', 0) for v in visits]
        )
        
        return {
            'patient_id': patient_id,
            'total_visits': len(visits),
            'first_visit': visits[0]['date'],
            'last_visit': visits[-1]['date'],
            'mmse_trend': mmse_trend,
            'nwbv_trend': nwbv_trend,
            'risk_trend': risk_trend,
            'progression_rate': self._calculate_progression_rate(visits),
            'recommendations': self._get_progression_recommendations(mmse_trend, nwbv_trend)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend from values."""
        if len(values) < 2:
            return {'direction': 'stable', 'rate': 0}
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            direction = 'improving'
        elif slope < -0.1:
            direction = 'declining'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'rate': round(slope, 4),
            'change': round(values[-1] - values[0], 2)
        }
    
    def _calculate_progression_rate(self, visits: List[Dict]) -> float:
        """Calculate overall progression rate."""
        if len(visits) < 2:
            return 0.0
        
        first_risk = visits[0]['prediction'].get('probability_demented', 0)
        last_risk = visits[-1]['prediction'].get('probability_demented', 0)
        
        days_diff = (datetime.fromisoformat(visits[-1]['date']) - 
                    datetime.fromisoformat(visits[0]['date'])).days
        
        if days_diff == 0:
            return 0.0
        
        rate = (last_risk - first_risk) / days_diff * 365  # per year
        return round(rate, 4)
    
    def _get_progression_recommendations(self, mmse_trend: Dict, nwbv_trend: Dict) -> List[str]:
        """Get recommendations based on progression."""
        recommendations = []
        
        if mmse_trend['direction'] == 'declining':
            recommendations.append("MMSE declining - consider neuropsychological evaluation")
        
        if nwbv_trend['direction'] == 'declining':
            recommendations.append("Brain volume decreasing - consider follow-up MRI")
        
        if mmse_trend['direction'] == 'declining' and nwbv_trend['direction'] == 'declining':
            recommendations.append("Multiple biomarkers declining - consider treatment intervention")
        
        if not recommendations:
            recommendations.append("Stable biomarkers - continue routine monitoring")
        
        return recommendations
    
    def load_history(self) -> Dict:
        """Load patient history."""
        if Path(self.history_file).exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_history(self, history: Dict):
        """Save patient history."""
        Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)


class ClinicalDecisionSupport:
    """Provide clinical decision support recommendations."""
    
    def __init__(self):
        """Initialize clinical decision support."""
        self.risk_scorer = RiskScorer()
    
    def get_recommendations(self, features: Dict, prediction: Dict) -> Dict:
        """
        Get comprehensive clinical recommendations.
        
        Args:
            features: Patient features
            prediction: Model prediction
            
        Returns:
            Clinical recommendations
        """
        risk_assessment = self.risk_scorer.calculate_risk_score(features)
        recommendations = []
        urgency = "routine"
        
        # Based on prediction
        if prediction.get('probability_demented', 0) > 0.7:
            recommendations.append({
                'type': 'diagnostic',
                'priority': 'high',
                'text': 'High probability of dementia detected. Recommend comprehensive evaluation including neuropsychological testing.'
            })
            urgency = "urgent"
        elif prediction.get('probability_demented', 0) > 0.5:
            recommendations.append({
                'type': 'monitoring',
                'priority': 'medium',
                'text': 'Moderate probability of dementia. Recommend close monitoring and follow-up in 3-6 months.'
            })
            urgency = "moderate"
        
        # Based on MMSE
        mmse = features.get('MMSE', 30)
        if mmse < 24:
            recommendations.append({
                'type': 'cognitive',
                'priority': 'high',
                'text': f'MMSE score ({mmse}) indicates cognitive impairment. Recommend detailed cognitive assessment.'
            })
            urgency = "urgent" if urgency != "urgent" else urgency
        
        # Based on brain volume
        nwbv = features.get('nWBV', 1.0)
        if nwbv < 0.70:
            recommendations.append({
                'type': 'imaging',
                'priority': 'medium',
                'text': 'Reduced normalized whole brain volume detected. Consider follow-up MRI in 6-12 months.'
            })
        
        # Based on age
        age = features.get('Age', 0)
        if age > 80:
            recommendations.append({
                'type': 'screening',
                'priority': 'low',
                'text': 'Advanced age - recommend annual cognitive screening.'
            })
        
        return {
            'recommendations': recommendations,
            'urgency': urgency,
            'risk_assessment': risk_assessment,
            'next_steps': self._get_next_steps(urgency, recommendations)
        }
    
    def _get_next_steps(self, urgency: str, recommendations: List[Dict]) -> List[str]:
        """Get next steps based on urgency and recommendations."""
        steps = []
        
        if urgency == "urgent":
            steps.append("Schedule comprehensive evaluation within 2 weeks")
            steps.append("Consider referral to neurologist or geriatrician")
        elif urgency == "moderate":
            steps.append("Schedule follow-up appointment in 3-6 months")
            steps.append("Continue routine monitoring")
        else:
            steps.append("Continue routine screening")
            steps.append("Reassess in 12 months")
        
        return steps


class ReportGenerator:
    """Generate clinical reports in PDF format."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_report(self, patient_info: Dict, features: Dict, prediction: Dict, 
                       risk_assessment: Dict, recommendations: Dict) -> str:
        """
        Generate comprehensive clinical report.
        
        Args:
            patient_info: Patient demographics
            features: Clinical features
            prediction: Model prediction
            risk_assessment: Risk scoring results
            recommendations: Clinical recommendations
            
        Returns:
            Path to generated PDF report
        """
        # This would use reportlab or similar to generate PDF
        # For now, return JSON representation
        report = {
            'patient_info': patient_info,
            'assessment_date': datetime.now().strftime('%Y-%m-%d'),
            'clinical_features': features,
            'prediction': prediction,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'disclaimer': 'This report is for research purposes only and should not be used for clinical decision-making.'
        }
        
        report_path = f"reports/report_{patient_info.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d')}.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path

