"""
Real-time Performance Tracking for TripFix
Provides comprehensive performance monitoring, metrics collection,
and system health tracking with background aggregation.
"""

import time
import threading
import json
import sqlite3
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import functools
import os
from pathlib import Path


@dataclass
class PerformanceMetric:
    """Represents a single performance metric"""
    timestamp: datetime
    component: str
    operation: str
    duration: float
    success: bool
    metadata: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    avg_response_time: float
    requests_per_minute: float
    error_rate: float
    active_sessions: int
    memory_usage: float
    cpu_usage: float


class PerformanceDatabase:
    """SQLite database for storing performance metrics"""
    
    def __init__(self, db_path: str = "data/performance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the performance database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                component TEXT,
                operation TEXT,
                duration REAL,
                success BOOLEAN,
                metadata TEXT,
                session_id TEXT
            )
        ''')
        
        # System health table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                avg_response_time REAL,
                requests_per_minute REAL,
                error_rate REAL,
                active_sessions INTEGER,
                memory_usage REAL,
                cpu_usage REAL
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_component ON performance_metrics(component)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp)')
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, component, operation, duration, success, metadata, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp.isoformat(),
            metric.component,
            metric.operation,
            metric.duration,
            metric.success,
            json.dumps(metric.metadata),
            metric.session_id
        ))
        
        conn.commit()
        conn.close()
    
    def store_health(self, health: SystemHealth):
        """Store system health metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_health 
            (timestamp, avg_response_time, requests_per_minute, error_rate, 
             active_sessions, memory_usage, cpu_usage)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            health.timestamp.isoformat(),
            health.avg_response_time,
            health.requests_per_minute,
            health.error_rate,
            health.active_sessions,
            health.memory_usage,
            health.cpu_usage
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   component: Optional[str] = None,
                   limit: int = 1000) -> List[PerformanceMetric]:
        """Retrieve performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM performance_metrics WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time.isoformat())
        
        if component:
            query += ' AND component = ?'
            params.append(component)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        metrics = []
        for row in rows:
            metrics.append(PerformanceMetric(
                timestamp=datetime.fromisoformat(row[1]),
                component=row[2],
                operation=row[3],
                duration=row[4],
                success=bool(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
                session_id=row[7]
            ))
        
        conn.close()
        return metrics
    
    def get_health_metrics(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 1000) -> List[SystemHealth]:
        """Retrieve system health metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM system_health WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time.isoformat())
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        health_metrics = []
        for row in rows:
            health_metrics.append(SystemHealth(
                timestamp=datetime.fromisoformat(row[1]),
                avg_response_time=row[2],
                requests_per_minute=row[3],
                error_rate=row[4],
                active_sessions=row[5],
                memory_usage=row[6],
                cpu_usage=row[7]
            ))
        
        conn.close()
        return health_metrics


class PerformanceTracker:
    """Real-time performance tracking and monitoring"""
    
    def __init__(self, db_path: str = "data/performance.db"):
        self.db = PerformanceDatabase(db_path)
        self.metrics_buffer = deque(maxlen=1000)  # In-memory buffer for recent metrics
        self.active_sessions = set()
        self.request_counts = deque(maxlen=60)  # Last 60 minutes
        self.error_counts = deque(maxlen=60)
        
        # Background aggregation
        self.aggregation_thread = None
        self.running = False
        self.start_background_aggregation()
    
    def start_background_aggregation(self):
        """Start background thread for metrics aggregation"""
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            return
        
        self.running = True
        self.aggregation_thread = threading.Thread(target=self._background_aggregation, daemon=True)
        self.aggregation_thread.start()
    
    def stop_background_aggregation(self):
        """Stop background aggregation thread"""
        self.running = False
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
    
    def _background_aggregation(self):
        """Background thread for aggregating and storing metrics"""
        while self.running:
            try:
                # Aggregate metrics every minute
                time.sleep(60)
                
                # Calculate system health metrics
                health = self._calculate_system_health()
                self.db.store_health(health)
                
                # Store buffered metrics
                while self.metrics_buffer:
                    metric = self.metrics_buffer.popleft()
                    self.db.store_metric(metric)
                
            except Exception as e:
                print(f"Error in background aggregation: {e}")
    
    def _calculate_system_health(self) -> SystemHealth:
        """Calculate current system health metrics"""
        now = datetime.now()
        
        # Calculate average response time from recent metrics
        recent_metrics = [m for m in self.metrics_buffer 
                         if (now - m.timestamp).total_seconds() < 300]  # Last 5 minutes
        
        avg_response_time = 0.0
        if recent_metrics:
            avg_response_time = statistics.mean([m.duration for m in recent_metrics])
        
        # Calculate requests per minute
        requests_per_minute = len(self.request_counts)
        
        # Calculate error rate
        error_rate = 0.0
        if self.request_counts:
            total_requests = sum(self.request_counts)
            total_errors = sum(self.error_counts)
            error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        
        # Get system resource usage (simplified)
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        
        return SystemHealth(
            timestamp=now,
            avg_response_time=avg_response_time,
            requests_per_minute=requests_per_minute,
            error_rate=error_rate,
            active_sessions=len(self.active_sessions),
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback to a simple estimation
            return 50.0  # Mock value
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            # Fallback to a simple estimation
            return 25.0  # Mock value
    
    def track_metric(self, 
                    component: str,
                    operation: str,
                    duration: float,
                    success: bool = True,
                    metadata: Optional[Dict[str, Any]] = None,
                    session_id: Optional[str] = None):
        """Track a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            duration=duration,
            success=success,
            metadata=metadata or {},
            session_id=session_id
        )
        
        # Add to buffer for real-time access
        self.metrics_buffer.append(metric)
        
        # Update request counts
        self.request_counts.append(1)
        if not success:
            self.error_counts.append(1)
        else:
            self.error_counts.append(0)
    
    def track_session_start(self, session_id: str):
        """Track session start"""
        self.active_sessions.add(session_id)
    
    def track_session_end(self, session_id: str):
        """Track session end"""
        self.active_sessions.discard(session_id)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        now = datetime.now()
        
        # Recent metrics (last 5 minutes)
        recent_metrics = [m for m in self.metrics_buffer 
                         if (now - m.timestamp).total_seconds() < 300]
        
        if not recent_metrics:
            return {
                "avg_response_time": 0.0,
                "requests_per_minute": 0.0,
                "error_rate": 0.0,
                "active_sessions": len(self.active_sessions),
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0
            }
        
        avg_response_time = statistics.mean([m.duration for m in recent_metrics])
        successful_requests = sum(1 for m in recent_metrics if m.success)
        failed_requests = len(recent_metrics) - successful_requests
        error_rate = failed_requests / len(recent_metrics) if recent_metrics else 0.0
        
        return {
            "avg_response_time": avg_response_time,
            "requests_per_minute": len(recent_metrics) / 5.0,  # Per minute
            "error_rate": error_rate,
            "active_sessions": len(self.active_sessions),
            "total_requests": len(recent_metrics),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests
        }
    
    def get_recent_performance(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent performance data for visualization"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        metrics = self.db.get_metrics(start_time, end_time, limit=1000)
        
        # Group by minute for visualization
        performance_data = defaultdict(list)
        for metric in metrics:
            minute_key = metric.timestamp.replace(second=0, microsecond=0)
            performance_data[minute_key].append(metric.duration)
        
        # Calculate average response time per minute
        result = []
        for timestamp, durations in performance_data.items():
            result.append({
                "timestamp": timestamp,
                "response_time": statistics.mean(durations),
                "request_count": len(durations)
            })
        
        return sorted(result, key=lambda x: x["timestamp"])
    
    def get_component_performance(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for a specific component"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.db.get_metrics(start_time, end_time, component=component)
        
        if not metrics:
            return {
                "total_requests": 0,
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0
            }
        
        durations = [m.duration for m in metrics]
        successful = [m for m in metrics if m.success]
        
        # Calculate percentiles
        durations_sorted = sorted(durations)
        p95_index = int(len(durations_sorted) * 0.95)
        p99_index = int(len(durations_sorted) * 0.99)
        
        return {
            "total_requests": len(metrics),
            "avg_response_time": statistics.mean(durations),
            "success_rate": len(successful) / len(metrics),
            "error_rate": 1.0 - (len(successful) / len(metrics)),
            "p95_response_time": durations_sorted[p95_index] if p95_index < len(durations_sorted) else durations_sorted[-1],
            "p99_response_time": durations_sorted[p99_index] if p99_index < len(durations_sorted) else durations_sorted[-1]
        }
    
    def get_system_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get system health history"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        return self.db.get_health_metrics(start_time, end_time)


def track_performance(component: str, operation: str = None):
    """Decorator for tracking function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get global performance tracker instance
            tracker = getattr(track_performance, '_tracker', None)
            if not tracker:
                return func(*args, **kwargs)
            
            operation_name = operation or func.__name__
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                metadata = {"error": error} if error else {}
                
                tracker.track_metric(
                    component=component,
                    operation=operation_name,
                    duration=duration,
                    success=success,
                    metadata=metadata
                )
        
        return wrapper
    return decorator


def track_session(func: Callable) -> Callable:
    """Decorator for tracking session-based operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get global performance tracker instance
        tracker = getattr(track_performance, '_tracker', None)
        if not tracker:
            return func(*args, **kwargs)
        
        # Extract session_id from arguments or kwargs
        session_id = None
        if args and hasattr(args[0], 'session_id'):
            session_id = args[0].session_id
        elif 'session_id' in kwargs:
            session_id = kwargs['session_id']
        
        if session_id:
            tracker.track_session_start(session_id)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if session_id:
                tracker.track_session_end(session_id)
    
    return wrapper


# Global performance tracker instance
_global_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
        track_performance._tracker = _global_tracker
    return _global_tracker

def set_performance_tracker(tracker: PerformanceTracker):
    """Set the global performance tracker instance"""
    global _global_tracker
    _global_tracker = tracker
    track_performance._tracker = tracker


# Example usage and testing
if __name__ == "__main__":
    # Initialize tracker
    tracker = PerformanceTracker()
    set_performance_tracker(tracker)
    
    # Example of using the decorator
    @track_performance("test_component", "example_operation")
    def example_function():
        time.sleep(0.1)  # Simulate work
        return "success"
    
    # Test the function
    result = example_function()
    print(f"Function result: {result}")
    
    # Get current metrics
    metrics = tracker.get_current_metrics()
    print(f"Current metrics: {metrics}")
    
    # Cleanup
    tracker.stop_background_aggregation()
