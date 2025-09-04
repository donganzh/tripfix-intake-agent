"""
TripFix Intake Dashboard
Displays all completed intake sessions in a comprehensive table format
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Import our database component
from utils.database import IntakeDatabase


class IntakeDashboard:
    """Dashboard for displaying completed intake sessions"""
    
    def __init__(self):
        self.database = IntakeDatabase()
    
    def render_header(self):
        """Render the dashboard header"""
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white; text-align: center;">
            <h1>📋 TripFix Intake Dashboard</h1>
            <p>View all completed intake sessions and their analysis results</p>
        </div>
        """, unsafe_allow_html=True)
    
    def format_compensation_amount(self, amount: Any) -> str:
        """Format compensation amount for display"""
        if amount is None or amount == 0:
            return "No compensation"
        try:
            return f"${float(amount):,.0f}"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_confidence(self, confidence: Any) -> str:
        """Format confidence score for display"""
        if confidence is None:
            return "N/A"
        try:
            return f"{float(confidence):.1%}"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_jurisdiction(self, jurisdiction: str) -> str:
        """Format jurisdiction with emoji"""
        if not jurisdiction:
            return "N/A"
        
        jurisdiction_map = {
            "APPR": "🇨🇦 APPR (Canada)",
            "EU261": "🇪🇺 EU261 (Europe)", 
            "NEITHER": "❌ No Coverage"
        }
        return jurisdiction_map.get(jurisdiction, jurisdiction)
    
    def format_status(self, status: str) -> str:
        """Format status with emoji and color"""
        if not status:
            return "N/A"
        
        status_map = {
            "in_progress": "🔄 In Progress",
            "eligibility_assessed": "✅ Assessed",
            "human_review_required": "👤 Human Review",
            "completed": "✅ Completed"
        }
        return status_map.get(status, status)
    
    def parse_eligibility_result(self, eligibility_result: str) -> Dict[str, Any]:
        """Parse eligibility result JSON"""
        if not eligibility_result:
            return {"eligible": False, "compensation_amount": 0, "reasoning": "No data"}
        
        try:
            return json.loads(eligibility_result)
        except (json.JSONDecodeError, TypeError):
            return {"eligible": False, "compensation_amount": 0, "reasoning": "Parse error"}
    
    def parse_flight_data(self, flight_data: str) -> Dict[str, Any]:
        """Parse flight data JSON"""
        if not flight_data:
            return {}
        
        try:
            return json.loads(flight_data)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def create_summary_stats(self, sessions: List[Dict[str, Any]]):
        """Create summary statistics"""
        if not sessions:
            return
        
        st.subheader("📊 Summary Statistics")
        
        # Calculate stats
        total_sessions = len(sessions)
        eligible_count = sum(1 for s in sessions if self.parse_eligibility_result(s.get('eligibility_result', '{}')).get('eligible', False))
        total_compensation = sum(float(s.get('compensation_amount', 0) or 0) for s in sessions)
        avg_confidence = sum(float(s.get('jurisdiction_confidence', 0) or 0) for s in sessions) / total_sessions if total_sessions > 0 else 0
        
        # Jurisdiction breakdown
        jurisdiction_counts = {}
        for session in sessions:
            jurisdiction = session.get('jurisdiction', 'Unknown')
            jurisdiction_counts[jurisdiction] = jurisdiction_counts.get(jurisdiction, 0) + 1
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", total_sessions)
        
        with col2:
            st.metric("Eligible Cases", f"{eligible_count} ({eligible_count/total_sessions:.1%})" if total_sessions > 0 else "0")
        
        with col3:
            st.metric("Total Compensation", f"${total_compensation:,.0f}")
        
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Jurisdiction breakdown
        st.write("**Jurisdiction Breakdown:**")
        jurisdiction_cols = st.columns(len(jurisdiction_counts))
        for i, (jurisdiction, count) in enumerate(jurisdiction_counts.items()):
            with jurisdiction_cols[i]:
                st.metric(
                    self.format_jurisdiction(jurisdiction),
                    f"{count} ({count/total_sessions:.1%})" if total_sessions > 0 else "0"
                )
    
    def render_sessions_table(self, sessions: List[Dict[str, Any]]):
        """Render the main sessions table"""
        if not sessions:
            st.warning("No completed intake sessions found.")
            return
        
        st.subheader("📋 Completed Intake Sessions")
        
        # Prepare data for display
        display_data = []
        for session in sessions:
            eligibility = self.parse_eligibility_result(session.get('eligibility_result', '{}'))
            flight_data = self.parse_flight_data(session.get('flight_data', '{}'))
            
            # Extract key flight information
            flight_info = []
            if flight_data.get('flight_numbers'):
                flight_info.append(f"Flight: {', '.join(flight_data['flight_numbers'])}")
            if flight_data.get('airlines'):
                flight_info.append(f"Airline: {', '.join(flight_data['airlines'])}")
            if flight_data.get('origin') and flight_data.get('destination'):
                flight_info.append(f"Route: {flight_data['origin']} → {flight_data['destination']}")
            
            display_data.append({
                "Session ID": session.get('id', 'N/A')[:8] + "...",
                "Created": session.get('created_at', 'N/A')[:10] if session.get('created_at') else 'N/A',
                "Status": self.format_status(session.get('status', '')),
                "Jurisdiction": self.format_jurisdiction(session.get('jurisdiction', '')),
                "Jurisdiction Confidence": self.format_confidence(session.get('jurisdiction_confidence')),
                "Eligible": "✅ Yes" if eligibility.get('eligible', False) else "❌ No",
                "Compensation": self.format_compensation_amount(eligibility.get('compensation_amount', session.get('compensation_amount'))),
                "Eligibility Confidence": self.format_confidence(session.get('eligibility_confidence')),
                "Risk Level": session.get('risk_level', 'N/A'),
                "Flight Info": " | ".join(flight_info) if flight_info else "N/A",
                "Handoff Reason": session.get('handoff_reason', 'N/A')[:50] + "..." if session.get('handoff_reason') and len(session.get('handoff_reason', '')) > 50 else session.get('handoff_reason', 'N/A')
            })
        
        # Create DataFrame
        df = pd.DataFrame(display_data)
        
        # Add filters
        st.write("**Filters:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(df['Status'].unique()))
        
        with col2:
            jurisdiction_filter = st.selectbox("Filter by Jurisdiction", ["All"] + list(df['Jurisdiction'].unique()))
        
        with col3:
            eligible_filter = st.selectbox("Filter by Eligibility", ["All", "Eligible", "Not Eligible"])
        
        # Apply filters
        filtered_df = df.copy()
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
        if jurisdiction_filter != "All":
            filtered_df = filtered_df[filtered_df['Jurisdiction'] == jurisdiction_filter]
        if eligible_filter == "Eligible":
            filtered_df = filtered_df[filtered_df['Eligible'] == "✅ Yes"]
        elif eligible_filter == "Not Eligible":
            filtered_df = filtered_df[filtered_df['Eligible'] == "❌ No"]
        
        # Display table
        st.write(f"Showing {len(filtered_df)} of {len(df)} sessions")
        
        # Style the dataframe
        def highlight_eligible(val):
            if val == "✅ Yes":
                return 'background-color: #d4edda'
            elif val == "❌ No":
                return 'background-color: #f8d7da'
            return ''
        
        def highlight_compensation(val):
            if val and val != "No compensation" and val != "N/A":
                return 'background-color: #d1ecf1'
            return ''
        
        styled_df = filtered_df.style.applymap(highlight_eligible, subset=['Eligible']).applymap(highlight_compensation, subset=['Compensation'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Export functionality
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"tripfix_intake_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def render_detailed_view(self, sessions: List[Dict[str, Any]]):
        """Render detailed view for selected session"""
        if not sessions:
            return
        
        st.subheader("🔍 Detailed Session View")
        
        # Session selector
        session_options = {f"{s['id'][:8]}... - {s.get('created_at', 'Unknown')[:10]}": s for s in sessions}
        selected_session_key = st.selectbox("Select a session to view details:", list(session_options.keys()))
        
        if selected_session_key:
            session = session_options[selected_session_key]
            
            # Display detailed information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session Information:**")
                st.write(f"• **ID:** {session.get('id', 'N/A')}")
                st.write(f"• **Created:** {session.get('created_at', 'N/A')}")
                st.write(f"• **Updated:** {session.get('updated_at', 'N/A')}")
                st.write(f"• **Status:** {self.format_status(session.get('status', ''))}")
                st.write(f"• **Completed:** {'✅ Yes' if session.get('completed') else '❌ No'}")
                
                st.write("**Jurisdiction Analysis:**")
                st.write(f"• **Jurisdiction:** {self.format_jurisdiction(session.get('jurisdiction', ''))}")
                st.write(f"• **Confidence:** {self.format_confidence(session.get('jurisdiction_confidence'))}")
            
            with col2:
                eligibility = self.parse_eligibility_result(session.get('eligibility_result', '{}'))
                st.write("**Eligibility Analysis:**")
                st.write(f"• **Eligible:** {'✅ Yes' if eligibility.get('eligible', False) else '❌ No'}")
                st.write(f"• **Compensation:** {self.format_compensation_amount(eligibility.get('compensation_amount', session.get('compensation_amount')))}")
                st.write(f"• **Confidence:** {self.format_confidence(session.get('eligibility_confidence'))}")
                st.write(f"• **Risk Level:** {session.get('risk_level', 'N/A')}")
                st.write(f"• **Handoff Priority:** {session.get('handoff_priority', 'N/A')}")
            
            # Flight data
            flight_data = self.parse_flight_data(session.get('flight_data', '{}'))
            if flight_data:
                st.write("**Flight Information:**")
                flight_col1, flight_col2 = st.columns(2)
                
                with flight_col1:
                    if flight_data.get('flight_numbers'):
                        st.write(f"• **Flight Numbers:** {', '.join(flight_data['flight_numbers'])}")
                    if flight_data.get('airlines'):
                        st.write(f"• **Airlines:** {', '.join(flight_data['airlines'])}")
                    if flight_data.get('dates'):
                        st.write(f"• **Dates:** {', '.join(flight_data['dates'])}")
                
                with flight_col2:
                    if flight_data.get('origin'):
                        st.write(f"• **Origin:** {flight_data['origin']}")
                    if flight_data.get('destination'):
                        st.write(f"• **Destination:** {flight_data['destination']}")
                    if flight_data.get('delay_length'):
                        st.write(f"• **Delay Length:** {flight_data['delay_length']}")
                    if flight_data.get('delay_reason'):
                        st.write(f"• **Delay Reason:** {flight_data['delay_reason']}")
            
            # Legal citations
            if session.get('legal_citations'):
                st.write("**Legal Citations:**")
                try:
                    citations = json.loads(session.get('legal_citations', '[]'))
                    for citation in citations:
                        st.write(f"• {citation}")
                except (json.JSONDecodeError, TypeError):
                    st.write(f"• {session.get('legal_citations', 'N/A')}")
            
            # Reasoning
            if eligibility.get('reasoning'):
                st.write("**Analysis Reasoning:**")
                st.write(eligibility['reasoning'])
            
            # Risk assessment
            if session.get('risk_assessment'):
                st.write("**Risk Assessment:**")
                try:
                    risk_assessment = json.loads(session.get('risk_assessment', '{}'))
                    if risk_assessment:
                        st.json(risk_assessment)
                except (json.JSONDecodeError, TypeError):
                    st.write(session.get('risk_assessment', 'N/A'))
    
    def render_dashboard(self):
        """Render the complete intake dashboard"""
        self.render_header()
        
        # Load completed sessions
        with st.spinner("Loading completed intake sessions..."):
            sessions = self.database.get_completed_sessions()
        
        if sessions:
            self.create_summary_stats(sessions)
            self.render_sessions_table(sessions)
            self.render_detailed_view(sessions)
        else:
            st.info("No completed intake sessions found. Complete some intake sessions to see them here.")
            
            # Show some sample data or instructions
            st.subheader("💡 How to create intake sessions:")
            st.write("""
            1. Go to the main TripFix application
            2. Start a new session and provide flight delay information
            3. Complete the intake process
            4. Return here to view the completed sessions
            """)


def main():
    """Main dashboard function"""
    # Page configuration
    st.set_page_config(
        page_title="TripFix Intake Dashboard",
        page_icon="📋",
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
    
    .session-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .eligible-session {
        border-left: 4px solid #28a745;
    }
    
    .ineligible-session {
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and render dashboard
    dashboard = IntakeDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
