"""
Beautiful Streamlit Evaluation Dashboard for TripFix
Provides comprehensive performance monitoring, metrics visualization,
and real-time system health monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

# Import our evaluation components
from utils.agent_evaluator import AgentEvaluator, EvaluationMetrics, GoldenTestDataset
from utils.performance_tracker import PerformanceTracker


class EvaluationDashboard:
    """Beautiful evaluation dashboard with comprehensive metrics"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.evaluator = None
        
    def render_header(self):
        """Render the dashboard header"""
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white; text-align: center;">
            <h1>📊 TripFix Agent Evaluation Dashboard</h1>
            <p>Comprehensive performance monitoring and system health analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_quick_stats(self, metrics: EvaluationMetrics):
        """Render quick statistics overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 Jurisdiction Accuracy",
                value=f"{metrics.jurisdiction_accuracy:.1%}",
                delta=f"{metrics.jurisdiction_accuracy - 0.85:.1%}" if metrics.jurisdiction_accuracy > 0.85 else None
            )
        
        with col2:
            st.metric(
                label="⚖️ Eligibility Accuracy", 
                value=f"{metrics.eligibility_accuracy:.1%}",
                delta=f"{metrics.eligibility_accuracy - 0.80:.1%}" if metrics.eligibility_accuracy > 0.80 else None
            )
        
        with col3:
            st.metric(
                label="🔄 Handoff F1 Score",
                value=f"{metrics.handoff_f1:.3f}",
                delta=f"{metrics.handoff_f1 - 0.70:.3f}" if metrics.handoff_f1 > 0.70 else None
            )
        
        with col4:
            st.metric(
                label="⏱️ Avg Processing Time",
                value=f"{metrics.average_processing_time:.1f}s",
                delta=f"{metrics.average_processing_time - 5.0:.1f}s" if metrics.average_processing_time < 5.0 else None
            )
    
    def render_performance_overview(self, metrics: EvaluationMetrics):
        """Render performance overview charts"""
        st.subheader("📈 Performance Overview")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy by Component", "Performance by Difficulty", 
                          "Jurisdiction Accuracy", "Confidence Distribution"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Accuracy by component
        components = ["Jurisdiction", "Eligibility", "Handoff"]
        accuracies = [metrics.jurisdiction_accuracy, metrics.eligibility_accuracy, metrics.handoff_f1]
        
        fig.add_trace(
            go.Bar(x=components, y=accuracies, name="Accuracy", 
                   marker_color=['#28a745', '#ffc107', '#dc3545']),
            row=1, col=1
        )
        
        # Performance by difficulty
        difficulties = list(metrics.performance_by_difficulty.keys())
        difficulty_scores = list(metrics.performance_by_difficulty.values())
        
        fig.add_trace(
            go.Bar(x=difficulties, y=difficulty_scores, name="Difficulty Performance",
                   marker_color=['#28a745', '#ffc107', '#dc3545']),
            row=1, col=2
        )
        
        # Jurisdiction accuracy by type
        jurisdictions = list(metrics.jurisdiction_accuracy_by_type.keys())
        jurisdiction_scores = list(metrics.jurisdiction_accuracy_by_type.values())
        
        fig.add_trace(
            go.Bar(x=jurisdictions, y=jurisdiction_scores, name="Jurisdiction Accuracy",
                   marker_color=['#007bff', '#6f42c1', '#6c757d']),
            row=2, col=1
        )
        
        # Confidence distribution
        confidence_scores = metrics.confidence_distribution.get("jurisdiction_confidence", [])
        if confidence_scores:
            fig.add_trace(
                go.Histogram(x=confidence_scores, name="Confidence Distribution",
                           marker_color='#17a2b8', nbinsx=20),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Performance Analytics")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_confidence_calibration(self, metrics: EvaluationMetrics):
        """Render confidence calibration analysis"""
        st.subheader("🎯 Confidence Calibration Analysis")
        
        # Create calibration plot
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
        accuracies = []
        confidences = []
        counts = []
        
        for bin_min, bin_max in bins:
            bin_results = [r for r in self.evaluator.results if bin_min <= r.jurisdiction_confidence < bin_max]
            if bin_results:
                accuracy = sum(1 for r in bin_results if r.jurisdiction_correct) / len(bin_results)
                confidence = sum(r.jurisdiction_confidence for r in bin_results) / len(bin_results)
                accuracies.append(accuracy)
                confidences.append(confidence)
                counts.append(len(bin_results))
            else:
                accuracies.append(0)
                confidences.append((bin_min + bin_max) / 2)
                counts.append(0)
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=confidences, y=accuracies,
            mode='markers+lines',
            name='Actual Calibration',
            marker=dict(size=counts, sizemode='diameter', sizeref=2, 
                       color=accuracies, colorscale='RdYlGn', showscale=True),
            text=[f"Count: {count}" for count in counts],
            hovertemplate="Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br>%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Confidence Calibration Plot",
            xaxis_title="Confidence Score",
            yaxis_title="Accuracy",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calibration metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Calibration Error", f"{metrics.confidence_calibration_error:.3f}")
        with col2:
            st.metric("Total Tests", f"{metrics.total_tests}")
        with col3:
            st.metric("Error Rate", f"{metrics.error_rate:.1%}")
    
    def render_retrieval_performance(self):
        """Render vector search and retrieval performance"""
        st.subheader("🔍 Retrieval Performance")
        
        # Mock retrieval metrics (in real implementation, these would come from vector store)
        retrieval_metrics = {
            "Average Response Time": 0.8,
            "Top-5 Accuracy": 0.92,
            "Top-10 Accuracy": 0.95,
            "Relevance Score": 0.88
        }
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_data = list(retrieval_metrics.items())
        
        for i, (metric, value) in enumerate(metrics_data):
            with [col1, col2, col3, col4][i]:
                if "Accuracy" in metric or "Score" in metric:
                    st.metric(metric, f"{value:.1%}")
                else:
                    st.metric(metric, f"{value:.1f}s")
        
        # Retrieval performance over time (mock data)
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        accuracy_data = [0.88 + (i * 0.01) + (0.02 * (i % 2)) for i in range(len(dates))]
        
        fig = px.line(
            x=dates, y=accuracy_data,
            title="Retrieval Accuracy Over Time",
            labels={"x": "Date", "y": "Accuracy"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_live_monitoring(self):
        """Render real-time system health monitoring"""
        st.subheader("⚡ Live System Monitoring")
        
        # Get current performance data
        current_metrics = self.performance_tracker.get_current_metrics()
        
        # System health indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_status = "🟢 Healthy" if current_metrics["avg_response_time"] < 5.0 else "🟡 Slow" if current_metrics["avg_response_time"] < 10.0 else "🔴 Critical"
            st.metric("System Health", health_status)
        
        with col2:
            st.metric("Active Sessions", current_metrics["active_sessions"])
        
        with col3:
            st.metric("Requests/min", current_metrics["requests_per_minute"])
        
        with col4:
            st.metric("Error Rate", f"{current_metrics['error_rate']:.1%}")
        
        # Real-time performance chart
        performance_data = self.performance_tracker.get_recent_performance()
        if performance_data:
            df = pd.DataFrame(performance_data)
            fig = px.line(
                df, x='timestamp', y='response_time',
                title="Response Time Over Last Hour",
                labels={"response_time": "Response Time (s)", "timestamp": "Time"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_test_case_analysis(self):
        """Render detailed test case analysis"""
        st.subheader("🧪 Test Case Analysis")
        
        if not self.evaluator or not self.evaluator.results:
            st.warning("No evaluation results available. Run an evaluation first.")
            return
        
        # Create results dataframe
        results_data = []
        test_case_map = {tc.id: tc for tc in self.evaluator.test_dataset.get_all_test_cases()}
        
        for result in self.evaluator.results:
            test_case = test_case_map.get(result.test_case_id)
            if test_case:
                results_data.append({
                    "Test Case": test_case.name,
                    "Difficulty": test_case.difficulty,
                    "Expected Jurisdiction": test_case.expected_jurisdiction,
                    "Actual Jurisdiction": result.actual_jurisdiction,
                    "Jurisdiction Correct": result.jurisdiction_correct,
                    "Expected Eligible": test_case.expected_eligible,
                    "Actual Eligible": result.actual_eligible,
                    "Eligibility Correct": result.eligibility_correct,
                    "Expected Handoff": test_case.expected_handoff,
                    "Actual Handoff": result.actual_handoff,
                    "Handoff Correct": result.handoff_correct,
                    "Jurisdiction Confidence": result.jurisdiction_confidence,
                    "Processing Time": result.processing_time,
                    "Error": result.error_message is not None
                })
        
        df = pd.DataFrame(results_data)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Accuracy by Difficulty**")
            difficulty_accuracy = df.groupby('Difficulty').agg({
                'Jurisdiction Correct': 'mean',
                'Eligibility Correct': 'mean',
                'Handoff Correct': 'mean'
            }).round(3)
            st.dataframe(difficulty_accuracy)
        
        with col2:
            st.write("**Performance by Jurisdiction**")
            jurisdiction_accuracy = df.groupby('Expected Jurisdiction').agg({
                'Jurisdiction Correct': 'mean',
                'Processing Time': 'mean'
            }).round(3)
            st.dataframe(jurisdiction_accuracy)
        
        # Detailed results table
        st.write("**Detailed Results**")
        
        # Add color coding for correct/incorrect
        def highlight_correct(val):
            if isinstance(val, bool):
                return 'background-color: #d4edda' if val else 'background-color: #f8d7da'
            return ''
        
        styled_df = df.style.applymap(highlight_correct, subset=['Jurisdiction Correct', 'Eligibility Correct', 'Handoff Correct'])
        st.dataframe(styled_df, use_container_width=True)
    
    def render_evaluation_controls(self):
        """Render evaluation controls and actions"""
        st.subheader("🎮 Evaluation Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Run Full Evaluation", type="primary"):
                with st.spinner("Running comprehensive evaluation..."):
                    self.run_evaluation()
        
        with col2:
            difficulty = st.selectbox("Quick Evaluation", ["easy", "medium", "hard"])
            if st.button("⚡ Quick Evaluation"):
                with st.spinner(f"Running {difficulty} evaluation..."):
                    self.run_quick_evaluation(difficulty)
        
        with col3:
            if st.button("📊 Load Previous Results"):
                self.load_previous_results()
        
        # Evaluation options
        st.write("**Evaluation Options**")
        col1, col2 = st.columns(2)
        
        with col1:
            include_edge_cases = st.checkbox("Include Edge Cases", value=True)
            include_confidence_tests = st.checkbox("Include Confidence Tests", value=True)
        
        with col2:
            save_results = st.checkbox("Save Results", value=True)
            generate_report = st.checkbox("Generate Report", value=True)
    
    def run_evaluation(self):
        """Run full evaluation suite"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return
            
            self.evaluator = AgentEvaluator(openai_api_key)
            
            # Run evaluation asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.evaluator.evaluate_all_cases())
            
            metrics = self.evaluator.calculate_metrics()
            
            st.success(f"Evaluation completed! Processed {len(results)} test cases.")
            st.session_state.evaluation_metrics = metrics
            st.session_state.evaluation_results = results
            
            # Save results if requested
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_results_{timestamp}.json"
                self.evaluator.save_results(filename)
                st.success(f"Results saved to {filename}")
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
    
    def run_quick_evaluation(self, difficulty: str):
        """Run quick evaluation on subset"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("OpenAI API key not found.")
                return
            
            self.evaluator = AgentEvaluator(openai_api_key)
            test_cases = self.evaluator.test_dataset.get_test_cases_by_difficulty(difficulty)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.evaluator.evaluate_all_cases(test_cases))
            
            metrics = self.evaluator.calculate_metrics()
            
            st.success(f"Quick evaluation completed! Processed {len(results)} {difficulty} test cases.")
            st.session_state.evaluation_metrics = metrics
            st.session_state.evaluation_results = results
            
        except Exception as e:
            st.error(f"Quick evaluation failed: {str(e)}")
    
    def load_previous_results(self):
        """Load previous evaluation results"""
        # In a real implementation, this would load from a file picker or database
        st.info("Load previous results functionality would be implemented here.")
    
    def render_dashboard(self):
        """Render the complete evaluation dashboard"""
        self.render_header()
        
        # Check if we have evaluation results
        metrics = st.session_state.get('evaluation_metrics')
        
        if metrics:
            self.render_quick_stats(metrics)
            self.render_performance_overview(metrics)
            self.render_confidence_calibration(metrics)
            self.render_test_case_analysis()
        else:
            st.info("No evaluation results available. Run an evaluation to see metrics.")
        
        # Always show these sections
        self.render_retrieval_performance()
        self.render_live_monitoring()
        self.render_evaluation_controls()


def main():
    """Main dashboard function"""
    # Page configuration
    st.set_page_config(
        page_title="TripFix Evaluation Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
    }
    
    .success-metric {
        border-left-color: #28a745;
    }
    
    .warning-metric {
        border-left-color: #ffc107;
    }
    
    .danger-metric {
        border-left-color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and render dashboard
    dashboard = EvaluationDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
