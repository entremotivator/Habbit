import streamlit as st
import pandas as pd
import datetime as dt
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import calendar

# Optional Google Sheets libraries (imported in functions to avoid errors)
# import gspread
# from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Content Habit Tracker Pro (90-Day)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': "# Content Habit Tracker Pro\nTrack your content creation journey over 90 days!"
    }
)

# ---------------------------
# Configuration & Constants
# ---------------------------
DEFAULT_SPREADSHEET_ID = "YOUR_SPREADSHEET_ID_HERE"
CHECKLIST_SHEET_NAME = "Checklist"
ACTIVITY_LOG_SHEET = "ActivityLog"
ANALYTICS_SHEET = "Analytics"
GOALS_SHEET = "Goals"

# Content Platforms with detailed categories
PLATFORMS = {
    "Video Platforms": [
        "YouTube Long-form",
        "YouTube Shorts",
        "TikTok",
        "Instagram Reels",
        "Facebook Video"
    ],
    "Social Media": [
        "LinkedIn Post",
        "LinkedIn Story",
        "Instagram Feed",
        "Instagram Stories",
        "Facebook Personal",
        "Facebook Page",
        "Twitter/X"
    ],
    "Community & Monetization": [
        "Fanbase",
        "Chatter",
        "Discord Community",
        "Email Newsletter",
        "Patreon"
    ],
    "Content Hub": [
        "Blog Website",
        "Podcast",
        "Live Stream"
    ]
}

# Comprehensive task categories
TASK_CATEGORIES = {
    "Content Creation": [
        "🎥 Record Main Video Content",
        "🎬 Film B-Roll Footage",
        "📱 Record Short-form Content",
        "📸 Take Photos for Posts",
        "🎙️ Record Podcast/Audio",
        "✍️ Write Blog Post",
        "📝 Write Newsletter",
        "🎨 Create Graphics/Thumbnails"
    ],
    "Content Editing": [
        "🎞️ Edit Long-form Video (30+ min)",
        "📺 Edit Medium Video (5-30 min)",
        "⚡ Edit Short-form Video (<5 min)",
        "🎵 Add Music & Sound Effects",
        "🔊 Audio Post-Production",
        "🖼️ Photo Editing & Enhancement",
        "📋 Write Captions & Descriptions",
        "🏷️ Create Tags & Keywords"
    ],
    "Publishing & Distribution": [],  # Will be populated from PLATFORMS
    "Engagement & Community": [
        "💬 Respond to Comments",
        "👥 Engage with Other Creators",
        "📧 Answer DMs/Messages",
        "🤝 Network with Industry Peers",
        "🎯 Reach out to Collaborators",
        "❤️ Like & Share Others' Content"
    ],
    "Analytics & Strategy": [
        "📊 Review Platform Analytics",
        "📈 Track Key Metrics",
        "🔍 Research Trending Topics",
        "🎯 Plan Next Week's Content",
        "💡 Brainstorm Content Ideas",
        "🗓️ Schedule Future Posts",
        "💰 Review Monetization Metrics"
    ],
    "Learning & Development": [
        "📚 Study Successful Creators",
        "🎓 Take Online Course/Tutorial",
        "📖 Read Industry Articles/Books",
        "🎪 Practice New Skills",
        "🔧 Learn New Tools/Software",
        "📝 Document Lessons Learned"
    ],
    "Business & Admin": [
        "💼 Update Business Profiles",
        "📊 Financial Tracking",
        "📝 Content Calendar Planning",
        "🤝 Sponsor/Brand Outreach",
        "📋 Organize Content Assets",
        "🔒 Backup Content & Data"
    ]
}

# Populate publishing tasks from platforms
for category, platforms in PLATFORMS.items():
    for platform in platforms:
        TASK_CATEGORIES["Publishing & Distribution"].append(f"🚀 Post to {platform}")

# Flatten all tasks
ALL_TASKS = []
for category, tasks in TASK_CATEGORIES.items():
    ALL_TASKS.extend(tasks)

# Habit difficulty levels
DIFFICULTY_LEVELS = {
    "🟢 Easy (1-2 min)": ["💬 Respond to Comments", "❤️ Like & Share Others' Content", "📧 Answer DMs/Messages"],
    "🟡 Medium (5-15 min)": ["📱 Record Short-form Content", "⚡ Edit Short-form Video (<5 min)", "🚀 Post to Instagram Stories"],
    "🟠 Hard (30-60 min)": ["🎥 Record Main Video Content", "📺 Edit Medium Video (5-30 min)", "✍️ Write Blog Post"],
    "🔴 Very Hard (1+ hours)": ["🎞️ Edit Long-form Video (30+ min)", "🎙️ Record Podcast/Audio", "📊 Review Platform Analytics"]
}

# Weekly goals and milestones
MILESTONE_DAYS = [7, 14, 21, 30, 45, 60, 75, 90]
DEFAULT_WEEKLY_GOALS = {
    "Videos Published": 3,
    "Social Posts": 10,
    "Engagement Sessions": 7,
    "Learning Hours": 2,
    "New Connections": 5
}

# ---------------------------
# Data Models & State Management
# ---------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    if "df" not in st.session_state:
        st.session_state.df = create_empty_dataframe()
    if "goals" not in st.session_state:
        st.session_state.goals = DEFAULT_WEEKLY_GOALS.copy()
    if "custom_tasks" not in st.session_state:
        st.session_state.custom_tasks = []
    if "streak_data" not in st.session_state:
        st.session_state.streak_data = {}
    if "current_day" not in st.session_state:
        st.session_state.current_day = 1
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Daily Tracker"

def create_empty_dataframe() -> pd.DataFrame:
    """Create empty 90-day dataframe with all columns"""
    columns = ["Day", "Date", "Week"] + ALL_TASKS + [
        "Daily_Notes", "Energy_Level", "Mood", "Challenges", "Wins", 
        "Tomorrow_Priority", "Time_Spent_Minutes", "Last_Updated"
    ]
    
    data = []
    start_date = dt.date.today()
    
    for i in range(1, 91):
        current_date = start_date + dt.timedelta(days=i-1)
        week_num = ((i - 1) // 7) + 1
        
        row = {
            "Day": i,
            "Date": current_date.strftime("%Y-%m-%d"),
            "Week": week_num,
            "Daily_Notes": "",
            "Energy_Level": 5,
            "Mood": "😐 Neutral",
            "Challenges": "",
            "Wins": "",
            "Tomorrow_Priority": "",
            "Time_Spent_Minutes": 0,
            "Last_Updated": ""
        }
        
        # Initialize all tasks as False
        for task in ALL_TASKS:
            row[task] = False
            
        data.append(row)
    
    return pd.DataFrame(data, columns=columns)

def calculate_streaks(df: pd.DataFrame) -> Dict[str, int]:
    """Calculate current streaks for each task"""
    streaks = {}
    
    for task in ALL_TASKS:
        current_streak = 0
        for i in range(len(df)-1, -1, -1):  # Go backwards from most recent day
            if df.iloc[i][task]:
                current_streak += 1
            else:
                break
        streaks[task] = current_streak
    
    return streaks

def calculate_completion_rate(df: pd.DataFrame, days_back: int = 7) -> float:
    """Calculate completion rate for recent days"""
    recent_df = df.tail(days_back)
    total_tasks = len(ALL_TASKS) * len(recent_df)
    completed_tasks = sum([sum(recent_df[task]) for task in ALL_TASKS])
    return (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

def get_task_difficulty(task: str) -> str:
    """Get difficulty level for a task"""
    for difficulty, tasks in DIFFICULTY_LEVELS.items():
        if task in tasks:
            return difficulty
    return "🟡 Medium (5-15 min)"  # Default

# ---------------------------
# Google Sheets Integration
# ---------------------------
def get_gs_client(creds_info: dict):
    """Create Google Sheets client"""
    from google.oauth2.service_account import Credentials
    import gspread
    
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(credentials)

def sync_with_sheets(client, spreadsheet_id: str, df: pd.DataFrame, operation: str = "save"):
    """Sync data with Google Sheets"""
    try:
        sh = client.open_by_key(spreadsheet_id)
        
        # Ensure required sheets exist
        sheet_names = [ws.title for ws in sh.worksheets()]
        required_sheets = [CHECKLIST_SHEET_NAME, ACTIVITY_LOG_SHEET, ANALYTICS_SHEET, GOALS_SHEET]
        
        for sheet_name in required_sheets:
            if sheet_name not in sheet_names:
                sh.add_worksheet(title=sheet_name, rows=200, cols=50)
        
        if operation == "save":
            # Save checklist data
            ws_checklist = sh.worksheet(CHECKLIST_SHEET_NAME)
            ws_checklist.clear()
            
            # Convert DataFrame to values for sheets
            values = [df.columns.tolist()]
            for _, row in df.iterrows():
                row_values = []
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, bool):
                        row_values.append("TRUE" if val else "FALSE")
                    elif pd.isna(val):
                        row_values.append("")
                    else:
                        row_values.append(str(val))
                values.append(row_values)
            
            ws_checklist.update("A1", values)
            
            # Save analytics summary
            ws_analytics = sh.worksheet(ANALYTICS_SHEET)
            analytics_data = generate_analytics_summary(df)
            ws_analytics.clear()
            ws_analytics.update("A1", analytics_data)
            
            return True
            
        elif operation == "load":
            ws_checklist = sh.worksheet(CHECKLIST_SHEET_NAME)
            values = ws_checklist.get_all_values()
            
            if not values or len(values) < 2:
                return df
            
            # Convert back to DataFrame
            headers = values[0]
            data = values[1:]
            loaded_df = pd.DataFrame(data, columns=headers)
            
            # Convert boolean columns back
            for task in ALL_TASKS:
                if task in loaded_df.columns:
                    loaded_df[task] = loaded_df[task].str.upper() == "TRUE"
            
            # Convert numeric columns
            numeric_cols = ["Day", "Week", "Energy_Level", "Time_Spent_Minutes"]
            for col in numeric_cols:
                if col in loaded_df.columns:
                    loaded_df[col] = pd.to_numeric(loaded_df[col], errors='coerce').fillna(0)
            
            return loaded_df
            
    except Exception as e:
        st.error(f"Google Sheets sync error: {e}")
        return df if operation == "load" else False

def generate_analytics_summary(df: pd.DataFrame) -> List[List]:
    """Generate analytics summary for sheets export"""
    summary = [
        ["Metric", "Value", "Description"],
        ["Total Days Tracked", len(df[df["Last_Updated"] != ""]), "Days with recorded activity"],
        ["Average Completion Rate", f"{calculate_completion_rate(df, 90):.1f}%", "Overall task completion percentage"],
        ["Current Streak (Days)", calculate_longest_streak(df), "Consecutive days with activity"],
        ["Most Productive Day", get_most_productive_day(df), "Day with highest completion rate"],
        ["Favorite Task Category", get_favorite_category(df), "Most frequently completed category"],
        ["Average Energy Level", f"{df['Energy_Level'].mean():.1f}/10", "Self-reported energy levels"],
        ["Total Time Invested", f"{df['Time_Spent_Minutes'].sum()} minutes", "Total time spent on content creation"]
    ]
    return summary

def calculate_longest_streak(df: pd.DataFrame) -> int:
    """Calculate longest consecutive streak"""
    max_streak = 0
    current_streak = 0
    
    for _, row in df.iterrows():
        if any(row[task] for task in ALL_TASKS):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def get_most_productive_day(df: pd.DataFrame) -> str:
    """Find the most productive day"""
    max_completion = 0
    best_day = "Day 1"
    
    for _, row in df.iterrows():
        completion = sum(row[task] for task in ALL_TASKS)
        if completion > max_completion:
            max_completion = completion
            best_day = f"Day {row['Day']}"
    
    return best_day

def get_favorite_category(df: pd.DataFrame) -> str:
    """Find most frequently completed task category"""
    category_scores = defaultdict(int)
    
    for category, tasks in TASK_CATEGORIES.items():
        for task in tasks:
            if task in df.columns:
                category_scores[category] += df[task].sum()
    
    return max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "Content Creation"

# ---------------------------
# UI Components
# ---------------------------
def render_sidebar():
    """Render the main sidebar with navigation and settings"""
    with st.sidebar:
        st.title("🎯 Content Habit Tracker Pro")
        
        # Navigation
        st.session_state.view_mode = st.selectbox(
            "📱 Choose View",
            ["Daily Tracker", "Weekly Overview", "Analytics Dashboard", "Goals & Milestones", "Settings"]
        )
        
        st.divider()
        
        # Google Sheets Integration
        st.subheader("☁️ Cloud Sync")
        creds_file = st.file_uploader("Service Account JSON", type=["json"])
        spreadsheet_id = st.text_input("Spreadsheet ID", value=DEFAULT_SPREADSHEET_ID)
        
        col1, col2 = st.columns(2)
        sync_save = col1.button("💾 Save", help="Upload current data to Google Sheets")
        sync_load = col2.button("🔄 Load", help="Download data from Google Sheets")
        
        # Handle sync operations
        if (sync_save or sync_load) and creds_file:
            try:
                creds_info = json.loads(creds_file.read().decode("utf-8"))
                client = get_gs_client(creds_info)
                
                if sync_save:
                    success = sync_with_sheets(client, spreadsheet_id, st.session_state.df, "save")
                    if success:
                        st.success("✅ Data saved to Google Sheets!")
                
                if sync_load:
                    loaded_df = sync_with_sheets(client, spreadsheet_id, st.session_state.df, "load")
                    st.session_state.df = loaded_df
                    st.success("✅ Data loaded from Google Sheets!")
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Sync failed: {e}")
        
        st.divider()
        
        # Quick Stats
        st.subheader("📊 Quick Stats")
        completion_rate = calculate_completion_rate(st.session_state.df, 7)
        st.metric("7-Day Completion", f"{completion_rate:.1f}%")
        
        total_days = len(st.session_state.df[st.session_state.df["Last_Updated"] != ""])
        st.metric("Days Tracked", total_days)
        
        current_streak = calculate_longest_streak(st.session_state.df)
        st.metric("Current Streak", f"{current_streak} days")

def render_daily_tracker():
    """Render the main daily tracking interface"""
    st.header("📅 Daily Habit Tracker")
    
    # Day selection and navigation
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
    
    with col1:
        current_day = st.number_input(
            "Select Day",
            min_value=1,
            max_value=90,
            value=st.session_state.current_day,
            key="day_selector"
        )
        st.session_state.current_day = current_day
    
    with col2:
        if st.button("◀️ Prev") and current_day > 1:
            st.session_state.current_day = current_day - 1
            st.experimental_rerun()
    
    with col3:
        if st.button("Today"):
            # Calculate current day based on start date
            st.session_state.current_day = min(90, max(1, 
                (dt.date.today() - dt.date.today()).days + 1))
            st.experimental_rerun()
    
    with col4:
        if st.button("Next ▶️") and current_day < 90:
            st.session_state.current_day = current_day + 1
            st.experimental_rerun()
    
    # Get current row data
    row_idx = current_day - 1
    current_row = st.session_state.df.iloc[row_idx].copy()
    
    # Display day info
    with col5:
        day_date = dt.datetime.strptime(current_row["Date"], "%Y-%m-%d").strftime("%B %d, %Y")
        st.write(f"**{day_date}** (Week {current_row['Week']})")
    
    st.divider()
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("✅ Mark All Publishing Done"):
            for task in TASK_CATEGORIES["Publishing & Distribution"]:
                current_row[task] = True
    
    with action_col2:
        if st.button("🎬 Mark All Creation Done"):
            for task in TASK_CATEGORIES["Content Creation"]:
                current_row[task] = True
    
    with action_col3:
        if st.button("💬 Mark All Engagement Done"):
            for task in TASK_CATEGORIES["Engagement & Community"]:
                current_row[task] = True
    
    with action_col4:
        if st.button("🔄 Clear All Tasks"):
            for task in ALL_TASKS:
                current_row[task] = False
    
    # Task categories with collapsible sections
    st.subheader("✅ Daily Tasks")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_completed = st.checkbox("Show completed tasks", value=True)
    with filter_col2:
        show_uncompleted = st.checkbox("Show uncompleted tasks", value=True)
    
    changes_made = False
    
    for category, tasks in TASK_CATEGORIES.items():
        if not tasks:  # Skip empty categories
            continue
            
        # Calculate category progress
        completed_in_category = sum(current_row.get(task, False) for task in tasks)
        total_in_category = len(tasks)
        category_progress = completed_in_category / total_in_category if total_in_category > 0 else 0
        
        # Create expandable section for each category
        with st.expander(
            f"{category} ({completed_in_category}/{total_in_category} completed - {category_progress:.0%})",
            expanded=(category in ["Content Creation", "Publishing & Distribution"])
        ):
            cols = st.columns(2)
            
            for i, task in enumerate(tasks):
                if task not in st.session_state.df.columns:
                    continue
                    
                current_value = current_row.get(task, False)
                
                # Apply filters
                if not show_completed and current_value:
                    continue
                if not show_uncompleted and not current_value:
                    continue
                
                with cols[i % 2]:
                    # Add difficulty indicator
                    difficulty = get_task_difficulty(task)
                    difficulty_emoji = difficulty.split()[0]
                    
                    new_value = st.checkbox(
                        f"{task} {difficulty_emoji}",
                        value=current_value,
                        key=f"task_{current_day}_{task}",
                        help=f"Difficulty: {difficulty}"
                    )
                    
                    if new_value != current_value:
                        current_row[task] = new_value
                        changes_made = True
    
    st.divider()
    
    # Daily reflection section
    st.subheader("📝 Daily Reflection")
    
    ref_col1, ref_col2 = st.columns(2)
    
    with ref_col1:
        current_row["Energy_Level"] = st.slider(
            "Energy Level (1-10)",
            min_value=1,
            max_value=10,
            value=int(current_row.get("Energy_Level", 5)),
            key=f"energy_{current_day}"
        )
        
        current_row["Mood"] = st.selectbox(
            "Mood",
            ["😊 Great", "😌 Good", "😐 Neutral", "😔 Low", "😫 Exhausted"],
            index=["😊 Great", "😌 Good", "😐 Neutral", "😔 Low", "😫 Exhausted"].index(
                current_row.get("Mood", "😐 Neutral")
            ),
            key=f"mood_{current_day}"
        )
        
        current_row["Time_Spent_Minutes"] = st.number_input(
            "Time Spent (minutes)",
            min_value=0,
            max_value=1440,  # 24 hours
            value=int(current_row.get("Time_Spent_Minutes", 0)),
            key=f"time_{current_day}"
        )
    
    with ref_col2:
        current_row["Wins"] = st.text_area(
            "🏆 Today's Wins",
            value=str(current_row.get("Wins", "")),
            placeholder="What went well today?",
            key=f"wins_{current_day}"
        )
        
        current_row["Challenges"] = st.text_area(
            "🚧 Challenges Faced",
            value=str(current_row.get("Challenges", "")),
            placeholder="What was difficult?",
            key=f"challenges_{current_day}"
        )
    
    current_row["Daily_Notes"] = st.text_area(
        "📋 General Notes",
        value=str(current_row.get("Daily_Notes", "")),
        placeholder="Any other thoughts, ideas, or observations...",
        key=f"notes_{current_day}"
    )
    
    current_row["Tomorrow_Priority"] = st.text_input(
        "🎯 Tomorrow's Top Priority",
        value=str(current_row.get("Tomorrow_Priority", "")),
        placeholder="What's the most important thing to focus on tomorrow?",
        key=f"priority_{current_day}"
    )
    
    # Progress visualization
    st.divider()
    st.subheader("📊 Today's Progress")
    
    total_tasks = len(ALL_TASKS)
    completed_tasks = sum(current_row.get(task, False) for task in ALL_TASKS)
    progress = completed_tasks / total_tasks if total_tasks > 0 else 0
    
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        st.metric("Tasks Completed", f"{completed_tasks}/{total_tasks}")
    
    with progress_col2:
        st.metric("Completion Rate", f"{progress:.1%}")
    
    with progress_col3:
        time_spent = current_row.get("Time_Spent_Minutes", 0)
        hours = time_spent // 60
        minutes = time_spent % 60
        st.metric("Time Invested", f"{hours}h {minutes}m")
    
    # Progress bar
    st.progress(progress, text=f"Day {current_day} Progress: {progress:.1%}")
    
    # Save changes
    if changes_made or st.button("💾 Save Today's Progress", type="primary"):
        current_row["Last_Updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.df.iloc[row_idx] = current_row
        st.success("✅ Progress saved!")

def render_weekly_overview():
    """Render weekly overview with aggregated statistics"""
    st.header("📊 Weekly Overview")
    
    # Week selection
    week_num = st.selectbox(
        "Select Week",
        range(1, 14),  # 90 days = ~13 weeks
        index=min(12, ((st.session_state.current_day - 1) // 7))
    )
    
    # Get week data
    start_day = (week_num - 1) * 7 + 1
    end_day = min(90, week_num * 7)
    week_data = st.session_state.df.iloc[start_day-1:end_day].copy()
    
    st.subheader(f"Week {week_num} (Days {start_day}-{end_day})")
    
    # Week summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tasks_completed = sum([sum(week_data[task]) for task in ALL_TASKS])
        st.metric("Tasks Completed", total_tasks_completed)
    
    with col2:
        avg_completion = calculate_completion_rate(week_data, len(week_data))
        st.metric("Avg Completion Rate", f"{avg_completion:.1f}%")
    
    with col3:
        total_time = week_data["Time_Spent_Minutes"].sum()
        hours = total_time // 60
        st.metric("Total Time", f"{hours}h {total_time % 60}m")
    
    with col4:
        avg_energy = week_data["Energy_Level"].mean()
        st.metric("Avg Energy", f"{avg_energy:.1f}/10")
    
    # Daily breakdown chart
    st.subheader("📈 Daily Progress This Week")
    
    daily_progress = []
    for _, row in week_data.iterrows():
        completed = sum(row[task] for task in ALL_TASKS)
        total = len(ALL_TASKS)
        daily_progress.append({
            "Day": f"Day {row['Day']}",
            "Completion Rate": (completed / total) * 100,
            "Tasks Completed": completed,
            "Energy Level": row["Energy_Level"],
            "Time Spent (hours)": row["Time_Spent_Minutes"] / 60
        })
    
    if daily_progress:
        daily_df = pd.DataFrame(daily_progress)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Completion Rate', 'Tasks Completed', 'Energy Level', 'Time Spent'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Completion rate
        fig.add_trace(
            go.Bar(x=daily_df["Day"], y=daily_df["Completion Rate"], name="Completion Rate"),
            row=1, col=1
        )
        
        # Tasks completed
        fig.add_trace(
            go.Bar(x=daily_df["Day"], y=daily_df["Tasks Completed"], name="Tasks Completed"),
            row=1, col=2
        )
        
        # Energy level
        fig.add_trace(
            go.Scatter(x=daily_df["Day"], y=daily_df["Energy Level"], mode='lines+markers', name="Energy Level"),
            row=2, col=1
        )
        
        # Time spent
        fig.add_trace(
            go.Bar(x=daily_df["Day"], y=daily_df["Time Spent (hours)"], name="Time Spent"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Category performance
    st.subheader("🏆 Category Performance")
    
    category_performance = {}
    for category, tasks in TASK_CATEGORIES.items():
        if tasks:
            completed = sum([week_data[task].sum() for task in tasks if task in week_data.columns])
            total_possible = len(tasks) * len(week_data)
            percentage = (completed / total_possible * 100) if total_possible > 0 else 0
            category_performance[category] = {
                "completed": completed,
                "total": total_possible,
                "percentage": percentage
            }
    
    # Display category performance
    for category, perf in category_performance.items():
        st.write(f"**{category}**: {perf['completed']}/{perf['total']} ({perf['percentage']:.1f}%)")
        st.progress(perf['percentage'] / 100)
    
    # Week reflection summary
    st.subheader("💭 Week Reflection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏆 This Week's Wins:**")
        wins = week_data["Wins"].dropna()
        wins = wins[wins != ""]
        if len(wins) > 0:
            for win in wins:
                if win.strip():
                    st.write(f"• {win}")
        else:
            st.write("No wins recorded this week")
    
    with col2:
        st.write("**🚧 This Week's Challenges:**")
        challenges = week_data["Challenges"].dropna()
        challenges = challenges[challenges != ""]
        if len(challenges) > 0:
            for challenge in challenges:
                if challenge.strip():
                    st.write(f"• {challenge}")
        else:
            st.write("No challenges recorded this week")

def render_analytics_dashboard():
    """Render comprehensive analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Time range selector
    time_range = st.selectbox(
        "Select Time Range",
        ["Last 7 days", "Last 14 days", "Last 30 days", "All time (90 days)"]
    )
    
    days_back = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "All time (90 days)": 90}[time_range]
    analysis_df = st.session_state.df.tail(days_back)
    
    # Overall metrics
    st.subheader("🎯 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        total_completion = calculate_completion_rate(analysis_df, len(analysis_df))
        st.metric("Overall Completion", f"{total_completion:.1f}%")
    
    with kpi_col2:
        active_days = len(analysis_df[analysis_df["Last_Updated"] != ""])
        st.metric("Active Days", f"{active_days}/{len(analysis_df)}")
    
    with kpi_col3:
        current_streak = calculate_longest_streak(analysis_df)
        st.metric("Current Streak", f"{current_streak} days")
    
    with kpi_col4:
        total_time_hours = analysis_df["Time_Spent_Minutes"].sum() / 60
        st.metric("Total Time", f"{total_time_hours:.1f}h")
    
    with kpi_col5:
        avg_energy = analysis_df["Energy_Level"].mean()
        st.metric("Avg Energy", f"{avg_energy:.1f}/10")
    
    # Completion rate over time
    st.subheader("📈 Completion Rate Trend")
    
    daily_rates = []
    for _, row in analysis_df.iterrows():
        completed = sum(row[task] for task in ALL_TASKS)
        total = len(ALL_TASKS)
        rate = (completed / total) * 100
        daily_rates.append({
            "Day": row["Day"],
            "Date": row["Date"],
            "Completion Rate": rate,
            "Energy Level": row["Energy_Level"]
        })
    
    if daily_rates:
        trend_df = pd.DataFrame(daily_rates)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=trend_df["Date"],
                y=trend_df["Completion Rate"],
                mode='lines+markers',
                name='Completion Rate (%)',
                line=dict(color='blue', width=3)
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=trend_df["Date"],
                y=trend_df["Energy Level"],
                mode='lines+markers',
                name='Energy Level',
                line=dict(color='orange', width=2, dash='dash')
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Completion Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Energy Level (1-10)", secondary_y=True)
        fig.update_layout(title="Progress and Energy Correlation", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Task category analysis
    st.subheader("🏷️ Task Category Performance")
    
    category_stats = []
    for category, tasks in TASK_CATEGORIES.items():
        if tasks:
            completed_tasks = sum([analysis_df[task].sum() for task in tasks if task in analysis_df.columns])
            total_possible = len(tasks) * len(analysis_df)
            completion_rate = (completed_tasks / total_possible * 100) if total_possible > 0 else 0
            
            category_stats.append({
                "Category": category,
                "Completion Rate": completion_rate,
                "Tasks Completed": completed_tasks,
                "Total Possible": total_possible
            })
    
    if category_stats:
        cat_df = pd.DataFrame(category_stats)
        
        fig = px.bar(
            cat_df,
            x="Category",
            y="Completion Rate",
            title="Completion Rate by Category",
            labels={"Completion Rate": "Completion Rate (%)"},
            color="Completion Rate",
            color_continuous_scale="Viridis"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of daily activity
    st.subheader("🔥 Activity Heatmap")
    
    # Create weekly heatmap data
    heatmap_data = []
    for _, row in analysis_df.iterrows():
        day_of_week = pd.to_datetime(row["Date"]).day_name()
        week_num = row["Week"]
        completion_rate = sum(row[task] for task in ALL_TASKS) / len(ALL_TASKS) * 100
        
        heatmap_data.append({
            "Week": f"Week {week_num}",
            "Day": day_of_week,
            "Completion Rate": completion_rate
        })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index="Week", columns="Day", values="Completion Rate")
        
        # Reorder days
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_pivot = heatmap_pivot.reindex(columns=[d for d in day_order if d in heatmap_pivot.columns])
        
        fig = px.imshow(
            heatmap_pivot,
            title="Weekly Activity Heatmap",
            labels=dict(x="Day of Week", y="Week", color="Completion Rate (%)"),
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Streaks analysis
    st.subheader("🔥 Streak Analysis")
    
    streaks = calculate_streaks(analysis_df)
    top_streaks = sorted(streaks.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_streaks:
        streak_df = pd.DataFrame(top_streaks, columns=["Task", "Current Streak"])
        
        fig = px.bar(
            streak_df,
            x="Current Streak",
            y="Task",
            orientation='h',
            title="Top 10 Current Streaks",
            color="Current Streak",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time investment analysis
    st.subheader("⏰ Time Investment Analysis")
    
    time_col1, time_col2 = st.columns(2)
    
    with time_col1:
        # Daily time spent
        time_data = analysis_df[analysis_df["Time_Spent_Minutes"] > 0]
        if len(time_data) > 0:
            fig = px.line(
                time_data,
                x="Date",
                y="Time_Spent_Minutes",
                title="Daily Time Investment",
                labels={"Time_Spent_Minutes": "Minutes", "Date": "Date"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with time_col2:
        # Time vs completion correlation
        if len(analysis_df) > 0:
            completion_rates = []
            for _, row in analysis_df.iterrows():
                rate = sum(row[task] for task in ALL_TASKS) / len(ALL_TASKS) * 100
                completion_rates.append(rate)
            
            corr_data = pd.DataFrame({
                "Time Spent (hours)": analysis_df["Time_Spent_Minutes"] / 60,
                "Completion Rate (%)": completion_rates
            })
            
            fig = px.scatter(
                corr_data,
                x="Time Spent (hours)",
                y="Completion Rate (%)",
                title="Time vs Completion Correlation",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_goals_milestones():
    """Render goals and milestones tracking"""
    st.header("🎯 Goals & Milestones")
    
    # Weekly goals setting
    st.subheader("📋 Weekly Goals")
    
    goal_col1, goal_col2 = st.columns(2)
    
    with goal_col1:
        st.write("**Set Your Weekly Targets:**")
        for goal_name, default_value in DEFAULT_WEEKLY_GOALS.items():
            st.session_state.goals[goal_name] = st.number_input(
                goal_name,
                min_value=0,
                value=st.session_state.goals.get(goal_name, default_value),
                key=f"goal_{goal_name}"
            )
    
    with goal_col2:
        st.write("**Current Week Progress:**")
        current_week = ((st.session_state.current_day - 1) // 7) + 1
        start_day = (current_week - 1) * 7 + 1
        end_day = min(90, current_week * 7)
        week_data = st.session_state.df.iloc[start_day-1:end_day]
        
        # Calculate actual progress
        videos_published = sum([week_data[task].sum() for task in TASK_CATEGORIES["Publishing & Distribution"] 
                               if "YouTube" in task or "Video" in task])
        social_posts = sum([week_data[task].sum() for task in TASK_CATEGORIES["Publishing & Distribution"]
                           if any(platform in task for platform in ["LinkedIn", "Instagram", "Facebook", "Twitter"])])
        engagement_sessions = sum([week_data[task].sum() for task in TASK_CATEGORIES["Engagement & Community"]])
        learning_hours = week_data["Time_Spent_Minutes"].sum() / 60  # Approximate learning time
        
        # Display progress
        st.metric("Videos Published", f"{videos_published}/{st.session_state.goals['Videos Published']}")
        st.metric("Social Posts", f"{social_posts}/{st.session_state.goals['Social Posts']}")
        st.metric("Engagement Sessions", f"{engagement_sessions}/{st.session_state.goals['Engagement Sessions']}")
        st.metric("Learning Hours", f"{learning_hours:.1f}/{st.session_state.goals['Learning Hours']}")
    
    # Milestone tracking
    st.subheader("🏁 Milestone Tracking")
    
    milestone_progress = []
    for milestone_day in MILESTONE_DAYS:
        if st.session_state.current_day >= milestone_day:
            milestone_data = st.session_state.df.iloc[:milestone_day]
            completion_rate = calculate_completion_rate(milestone_data, milestone_day)
            status = "✅ Completed" if st.session_state.current_day >= milestone_day else "⏳ In Progress"
        else:
            completion_rate = 0
            status = "🔜 Upcoming"
        
        milestone_progress.append({
            "Milestone": f"Day {milestone_day}",
            "Status": status,
            "Completion Rate": f"{completion_rate:.1f}%",
            "Days Until": max(0, milestone_day - st.session_state.current_day)
        })
    
    milestone_df = pd.DataFrame(milestone_progress)
    st.dataframe(milestone_df, use_container_width=True)
    
    # Achievement badges
    st.subheader("🏆 Achievement Badges")
    
    achievements = []
    
    # Check various achievements
    total_completion = calculate_completion_rate(st.session_state.df, 90)
    current_streak = calculate_longest_streak(st.session_state.df)
    total_time = st.session_state.df["Time_Spent_Minutes"].sum() / 60
    
    if current_streak >= 7:
        achievements.append("🔥 Week Warrior - 7+ day streak")
    if current_streak >= 30:
        achievements.append("💪 Month Master - 30+ day streak")
    if total_completion >= 80:
        achievements.append("🎯 High Achiever - 80%+ completion rate")
    if total_time >= 100:
        achievements.append("⏰ Time Investor - 100+ hours invested")
    
    # Publishing achievements
    publishing_tasks = sum([st.session_state.df[task].sum() for task in TASK_CATEGORIES["Publishing & Distribution"]])
    if publishing_tasks >= 50:
        achievements.append("📢 Content Creator - 50+ posts published")
    if publishing_tasks >= 100:
        achievements.append("🚀 Publishing Pro - 100+ posts published")
    
    # Display achievements
    if achievements:
        achievement_cols = st.columns(min(3, len(achievements)))
        for i, achievement in enumerate(achievements):
            with achievement_cols[i % 3]:
                st.success(achievement)
    else:
        st.info("Keep going! Achievements will unlock as you progress.")

def render_settings():
    """Render settings and customization options"""
    st.header("⚙️ Settings & Customization")
    
    # Custom tasks
    st.subheader("➕ Custom Tasks")
    
    with st.expander("Add Custom Task"):
        custom_task_name = st.text_input("Task Name", placeholder="e.g., 🎨 Create Custom Thumbnail")
        custom_task_category = st.selectbox("Category", list(TASK_CATEGORIES.keys()))
        
        if st.button("Add Task") and custom_task_name:
            if custom_task_name not in ALL_TASKS:
                # Add to category
                TASK_CATEGORIES[custom_task_category].append(custom_task_name)
                ALL_TASKS.append(custom_task_name)
                
                # Add column to dataframe
                st.session_state.df[custom_task_name] = False
                st.session_state.custom_tasks.append(custom_task_name)
                st.success(f"Added '{custom_task_name}' to {custom_task_category}")
                st.experimental_rerun()
            else:
                st.error("Task already exists")
    
    # Display custom tasks
    if st.session_state.custom_tasks:
        st.write("**Your Custom Tasks:**")
        for task in st.session_state.custom_tasks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {task}")
            with col2:
                if st.button("Remove", key=f"remove_{task}"):
                    # Remove from everywhere
                    for category_tasks in TASK_CATEGORIES.values():
                        if task in category_tasks:
                            category_tasks.remove(task)
                    ALL_TASKS.remove(task)
                    st.session_state.custom_tasks.remove(task)
                    if task in st.session_state.df.columns:
                        st.session_state.df.drop(columns=[task], inplace=True)
                    st.experimental_rerun()
    
    st.divider()
    
    # Data management
    st.subheader("💾 Data Management")
    
    data_col1, data_col2, data_col3 = st.columns(3)
    
    with data_col1:
        if st.button("📥 Export All Data"):
            # Create comprehensive export
            export_data = st.session_state.df.copy()
            csv_data = export_data.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                "content_habit_tracker_full.csv",
                "text/csv"
            )
    
    with data_col2:
        uploaded_file = st.file_uploader("📤 Import Data", type=['csv'])
        if uploaded_file:
            try:
                imported_df = pd.read_csv(uploaded_file)
                # Validate and merge data
                st.session_state.df = imported_df
                st.success("Data imported successfully!")
            except Exception as e:
                st.error(f"Import failed: {e}")
    
    with data_col3:
        if st.button("🔄 Reset All Data", help="This will clear all your progress!"):
            if st.checkbox("I understand this will delete all my data"):
                st.session_state.df = create_empty_dataframe()
                st.session_state.custom_tasks = []
                st.success("All data has been reset!")
                st.experimental_rerun()
    
    st.divider()
    
    # App preferences
    st.subheader("🎨 App Preferences")
    
    pref_col1, pref_col2 = st.columns(2)
    
    with pref_col1:
        # Theme and display options
        st.write("**Display Options:**")
        show_progress_bars = st.checkbox("Show progress bars in categories", value=True)
        show_difficulty_indicators = st.checkbox("Show task difficulty indicators", value=True)
        compact_view = st.checkbox("Use compact task view", value=False)
    
    with pref_col2:
        # Notification preferences
        st.write("**Reminder Preferences:**")
        daily_reminder_time = st.time_input("Daily reminder time", value=dt.time(9, 0))
        weekend_reminders = st.checkbox("Include weekend reminders", value=True)
    
    # Save preferences
    if st.button("💾 Save Preferences"):
        # In a real app, these would be saved to user preferences
        st.success("Preferences saved!")

# ---------------------------
# Main App Logic
# ---------------------------
def main():
    """Main application logic"""
    initialize_session_state()
    render_sidebar()
    
    # Route to different views based on selection
    if st.session_state.view_mode == "Daily Tracker":
        render_daily_tracker()
    elif st.session_state.view_mode == "Weekly Overview":
        render_weekly_overview()
    elif st.session_state.view_mode == "Analytics Dashboard":
        render_analytics_dashboard()
    elif st.session_state.view_mode == "Goals & Milestones":
        render_goals_milestones()
    elif st.session_state.view_mode == "Settings":
        render_settings()

if __name__ == "__main__":
    main()
