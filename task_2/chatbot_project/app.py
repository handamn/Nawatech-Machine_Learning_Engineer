import streamlit as st
import logging
import os
import sys
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.chatbot import NawatechChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_streamlit_config():
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

#session_stat
def init_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'system_info' not in st.session_state:
        st.session_state.system_info = {}
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False

@st.cache_resource
def initialize_chatbot():
    try:
        # Validate configuration
        Config.validate_config()
        
        # Initialize chatbot
        chatbot = NawatechChatbot(Config)
        
        # Setup database with FAQ data
        faq_path = os.path.join("data", "FAQ_Nawa.csv")
        
        chatbot.setup_database(faq_path)
        
        return chatbot
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        st.error(f"Failed to initialize chatbot: {e}")
        return None


#Render sidebar with system information and controls
def render_sidebar():
    st.sidebar.header("🤖 Nawatech Advanced Chatbot")
    
    # System status
    if st.session_state.chatbot:
        system_info = st.session_state.chatbot.get_system_info()
        st.session_state.system_info = system_info
        
        status_color = "🟢" if system_info.get("status") == "healthy" else "🔴"
        st.sidebar.markdown(f"**Status:** {status_color} {system_info.get('status', 'Unknown')}")
        
        # Component status
        components = system_info.get("components", {})
        st.sidebar.markdown("**🔧 Components:**")
        for comp, status in components.items():
            if status == "connected" or status == "loaded" or status == "enabled":
                comp_color = "🟢"
            elif status == "disabled" or status == "not_configured":
                comp_color = "🟡"
            else:
                comp_color = "🔴"
            st.sidebar.markdown(f"- {comp.replace('_', ' ').title()}: {comp_color} {status}")
        
        # Advanced Features Status
        config_info = system_info.get("config", {})
        st.sidebar.markdown("**🚀 Advanced Features:**")
        st.sidebar.markdown(f"- Database: {config_info.get('database_type', 'single').title()}")
        st.sidebar.markdown(f"- Search: {config_info.get('search_type', 'semantic').title()}")
        st.sidebar.markdown(f"- Security: {'🟢 Enabled' if config_info.get('security_enabled') else '🔴 Disabled'}")
        st.sidebar.markdown(f"- Advanced Scoring: {'🟢 Enabled' if config_info.get('advanced_scoring') else '🔴 Disabled'}")
        
        # Database info
        primary_db_info = system_info.get("primary_database_info", {})
        if primary_db_info and not primary_db_info.get("error"):
            st.sidebar.markdown("**📊 Primary Database (Qdrant):**")
            st.sidebar.markdown(f"- Documents: {primary_db_info.get('points_count', 0)}")
            st.sidebar.markdown(f"- Status: {primary_db_info.get('status', 'unknown')}")
        
        secondary_db_info = system_info.get("secondary_database_info", {})
        if secondary_db_info and not secondary_db_info.get("error"):
            st.sidebar.markdown("**📊 Secondary Database (Pinecone):**")
            st.sidebar.markdown(f"- Documents: {secondary_db_info.get('points_count', 0)}")
            st.sidebar.markdown(f"- Status: {secondary_db_info.get('status', 'unknown')}")
        
        # Performance info
        performance = system_info.get("performance", {})
        st.sidebar.markdown("**⚡ Performance:**")
        st.sidebar.markdown(f"- Conversation: {performance.get('conversation_length', 0)} messages")
        st.sidebar.markdown(f"- Rate Limit: {performance.get('rate_limit', 'N/A')}")
    
    # Configuration toggles
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Configuration:**")
    
    # Search type selection
    search_options = ["semantic", "hybrid", "keyword"]
    current_search = getattr(Config, 'SEARCH_TYPE', 'hybrid')
    if st.sidebar.selectbox("Search Type:", search_options, 
                           index=search_options.index(current_search) if current_search in search_options else 0):
        st.sidebar.info("Search type changed (restart to apply)")
    
    # Database selection
    if st.session_state.chatbot and hasattr(st.session_state.chatbot, 'secondary_db') and st.session_state.chatbot.secondary_db:
        if st.sidebar.button("🔄 Switch to Pinecone"):
            result = st.session_state.chatbot.switch_database(use_secondary=True)
            if result["success"]:
                st.sidebar.success(result["message"])
            else:
                st.sidebar.warning(result["message"])
    
    # LLM model selection
    llm_model = st.sidebar.radio("LLM Model:", ["OpenAI", "Ollama"])
    ollama_model = None
    if llm_model == "Ollama":
        # Buat sementara instance Ollama untuk ambil daftar model
        from src.ollama_llm import OllamaLLM
        temp_ollama = OllamaLLM()
        models = temp_ollama.list_models()
        if models:
            ollama_model = st.sidebar.selectbox("Select Ollama Model:", models)
        else:
            st.sidebar.warning("No Ollama models found")

    if st.session_state.chatbot:
        if st.sidebar.button("🔄 Switch LLM Model"):
            st.session_state.chatbot.switch_llm_model(llm_model, ollama_model)
            st.sidebar.success(f"Switched to {llm_model} ({ollama_model or 'default'})")

    
    # Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎮 Controls:**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reset", help="Reset conversation"):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.reset_conversation()
            st.rerun()
    
    with col2:
        if st.button("🔧 Reinit", help="Reinitialize system"):
            st.cache_resource.clear()
            st.session_state.chatbot = None
            st.session_state.chatbot_initialized = False
            st.rerun()
    
    # Security info
    if st.session_state.chatbot:
        if st.sidebar.button("🔒 Security Info"):
            security_info = st.session_state.chatbot.get_client_security_info()
            st.sidebar.json(security_info)

#Render main chat interface with advanced features
def render_chat_interface():
    st.header("💬 Chat with Nawatech Advanced Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show advanced metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("📊 Advanced Response Analytics"):
                    metadata = message["metadata"]
                    
                    # Quality metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        overall_score = metadata.get('overall_score', 0)
                        quality_tier = metadata.get('quality_tier', 'unknown')
                        st.metric("Overall Quality", f"{overall_score:.3f}", delta=quality_tier.title())
                    with col2:
                        confidence = metadata.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.3f}")
                    with col3:
                        context_used = metadata.get('context_used', 0)
                        st.metric("Context Used", context_used)
                    with col4:
                        tokens_used = metadata.get('tokens_used', 0)
                        st.metric("Tokens Used", tokens_used)
                    
                    # Detailed quality scores
                    detailed_scores = metadata.get('detailed_scores', {})
                    if detailed_scores:
                        st.markdown("**🎯 Quality Breakdown:**")
                        score_cols = st.columns(3)
                        score_items = list(detailed_scores.items())
                        for i, (metric, score) in enumerate(score_items):
                            with score_cols[i % 3]:
                                # Format metric name
                                formatted_metric = metric.replace('_', ' ').title()
                                # Color based on score
                                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                                st.markdown(f"{color} **{formatted_metric}**: {score:.3f}")
                    
                    # Search and database info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔍 Search Info:**")
                        search_strategy = metadata.get('search_strategy', {})
                        st.markdown(f"- Type: {search_strategy.get('search_type', 'unknown')}")
                        st.markdown(f"- Top-K: {search_strategy.get('top_k', 'N/A')}")
                        if search_strategy.get('hybrid_enabled'):
                            st.markdown(f"- Semantic Weight: {search_strategy.get('semantic_weight', 0):.1f}")
                            st.markdown(f"- Keyword Weight: {search_strategy.get('keyword_weight', 0):.1f}")
                    
                    with col2:
                        st.markdown("**💾 Database Info:**")
                        database_info = metadata.get('database_used', {})
                        st.markdown(f"- Primary: {database_info.get('primary_db', 'unknown')}")
                        if database_info.get('secondary_db'):
                            st.markdown(f"- Secondary: {database_info.get('secondary_db')}")
                        st.markdown(f"- Dual Mode: {'Yes' if database_info.get('dual_mode') else 'No'}")
                    
                    # Security info
                    security_info = metadata.get('security_info', {})
                    if security_info:
                        st.markdown("**🔒 Security Status:**")
                        security_status = metadata.get('security_status', 'unknown')
                        status_color = "🟢" if security_status == "passed" else "🔴"
                        st.markdown(f"{status_color} Status: {security_status}")
                        
                        risk_level = security_info.get('injection_risk', 'unknown')
                        risk_color = "🟢" if risk_level == "low" else "🟡" if risk_level == "medium" else "🔴"
                        st.markdown(f"{risk_color} Injection Risk: {risk_level}")
                    
                    # Source references
                    source_references = metadata.get('source_references', [])
                    if source_references:
                        st.markdown("**📚 Source References:**")
                        for i, ref in enumerate(source_references, 1):
                            st.markdown(f"{i}. {ref}")
                    
                    # Recommendations
                    recommendations = metadata.get('recommendations', [])
                    if recommendations:
                        st.markdown("**💡 Recommendations:**")
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    
                    # Score explanation
                    explanation = metadata.get('score_explanation', '')
                    if explanation:
                        st.markdown("**📝 Score Explanation:**")
                        st.markdown(explanation)
                    
                    # Timestamp
                    timestamp = metadata.get('timestamp')
                    if timestamp:
                        import datetime
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        st.markdown(f"**⏰ Generated:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")

#Render advanced features panel
def render_advanced_features():
    st.markdown("---")
    
    with st.expander("🚀 Advanced Features & Testing"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Search Benchmark**")
            test_query = st.text_input("Test Query:", value="Apa itu Nawatech?")
            if st.button("Run Benchmark") and st.session_state.chatbot:
                with st.spinner("Running benchmark..."):
                    benchmark_results = st.session_state.chatbot.benchmark_search_methods(test_query)
                
                if "benchmark_results" in benchmark_results:
                    st.markdown("**Results:**")
                    for method, results in benchmark_results["benchmark_results"].items():
                        st.markdown(f"**{method.title()}:**")
                        st.markdown(f"- Results: {results['results_count']}")
                        st.markdown(f"- Avg Score: {results['avg_score']:.3f}")
                        st.markdown(f"- Time: {results['response_time']:.3f}s")
                        st.markdown("---")
        
        with col2:
            st.markdown("**📊 System Statistics**")
            if st.button("Show Stats") and st.session_state.chatbot:
                system_info = st.session_state.chatbot.get_system_info()
                
                # Security stats
                security_stats = system_info.get("security_stats", {})
                if security_stats:
                    st.markdown("**Security:**")
                    st.markdown(f"- Total Clients: {security_stats.get('total_clients', 0)}")
                    st.markdown(f"- Total Requests: {security_stats.get('total_requests', 0)}")
                    st.markdown(f"- Recent Requests: {security_stats.get('recent_requests', 0)}")
                
                # Hybrid search stats
                hybrid_stats = system_info.get("hybrid_search_stats", {})
                if hybrid_stats:
                    st.markdown("**Hybrid Search:**")
                    st.markdown(f"- Indexed Docs: {hybrid_stats.get('indexed_documents', 0)}")
                    st.markdown(f"- Semantic Weight: {hybrid_stats.get('semantic_weight', 0)}")
                    st.markdown(f"- Keyword Weight: {hybrid_stats.get('keyword_weight', 0)}")

#Render suggested questions with categories
def render_suggested_questions():
    if st.session_state.chatbot:
        suggestions = st.session_state.chatbot.get_suggested_questions()
        if suggestions:
            st.markdown("**💡 Suggested Questions:**")
            
            # Group suggestions into rows of 2
            for i in range(0, len(suggestions), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(suggestions):
                        if st.button(suggestions[i], key=f"suggestion_{i}"):
                            process_suggestion(suggestions[i])
                
                with col2:
                    if i + 1 < len(suggestions):
                        if st.button(suggestions[i + 1], key=f"suggestion_{i + 1}"):
                            process_suggestion(suggestions[i + 1])

#Process a suggested question
def process_suggestion(suggestion: str):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": suggestion})
    
    # Process with chatbot
    with st.spinner("Processing..."):
        response_data = st.session_state.chatbot.chat(suggestion, client_id="streamlit_user")
    
    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["response"],
        "metadata": response_data
    })
    
    st.rerun()

#Render suggested questions
def render_suggested_questions():
    if st.session_state.chatbot:
        suggestions = st.session_state.chatbot.get_suggested_questions()
        if suggestions:
            st.markdown("**💡 Suggested Questions:**")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with cols[i % len(cols)]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        # Add user message
                        st.session_state.messages.append({"role": "user", "content": suggestion})
                        
                        # Process with chatbot
                        with st.spinner("Thinking..."):
                            response_data = st.session_state.chatbot.chat(suggestion)
                        
                        # Add assistant response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_data["response"],
                            "metadata": response_data
                        })
                        
                        st.rerun()

def main():
    init_streamlit_config()
    init_session_state()
    
    # Initialize chatbot
    if not st.session_state.chatbot_initialized:
        with st.spinner("Initializing Nawatech Chatbot..."):
            chatbot = initialize_chatbot()
            if chatbot:
                st.session_state.chatbot = chatbot
                st.session_state.chatbot_initialized = True
                st.success("Chatbot initialized successfully!")
            else:
                st.error("Failed to initialize chatbot. Please check your configuration.")
                st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Render main interface
    if st.session_state.chatbot_initialized:
        render_chat_interface()
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about Nawatech..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_data = st.session_state.chatbot.chat(prompt)
                
                st.write(response_data["response"])
                
                # Show response metadata
                with st.expander("📊 Response Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quality Score", f"{response_data.get('quality_score', 0):.2f}")
                    with col2:
                        st.metric("Confidence", f"{response_data.get('confidence', 0):.2f}")
                    with col3:
                        st.metric("Context Used", response_data.get('context_used', 0))
                    
                    if response_data.get('source_references'):
                        st.markdown("**Source References:**")
                        for ref in response_data['source_references']:
                            st.markdown(f"- {ref}")
            
            # Add assistant response to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["response"],
                "metadata": response_data
            })
        
        # Render suggested questions if no messages
        if not st.session_state.messages:
            render_suggested_questions()
    
    # Footer
    st.markdown("---")
    st.markdown("**Nawatech FAQ Chatbot** - Developed by Hanif Adam")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()