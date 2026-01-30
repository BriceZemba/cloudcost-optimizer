"""
Data Generator for Cloud Cost Optimization
Generates realistic cloud usage and cost data for training and testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict


class CloudDataGenerator:
    """Generate realistic cloud usage and cost data"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_daily_data(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        base_cost: float = 1000.0,
        trend: float = 0.05,
        seasonality: bool = True,
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate daily cloud usage and cost data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            base_cost: Base monthly cost in dollars
            trend: Monthly growth trend (0.05 = 5% growth)
            seasonality: Whether to include weekly/monthly patterns
            noise_level: Amount of random noise (0-1)
        
        Returns:
            DataFrame with daily metrics
        """
        # Date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate time features
        day_of_week = dates.dayofweek.values
        day_of_month = dates.day.values
        month = dates.month.values
        
        # Base patterns
        # 1. Trend component (gradual increase)
        trend_component = np.linspace(0, trend * n_days / 30, n_days)
        
        # 2. Seasonal components
        if seasonality:
            # Weekly pattern (lower on weekends)
            weekly_pattern = np.where(
                day_of_week < 5,  # Monday-Friday
                1.0,
                0.7  # Weekend
            )
            
            # Monthly pattern (peak mid-month)
            monthly_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_month / 30)
            
            # Yearly pattern (holiday season variations)
            yearly_pattern = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12)
        else:
            weekly_pattern = np.ones(n_days)
            monthly_pattern = np.ones(n_days)
            yearly_pattern = np.ones(n_days)
        
        # 3. Random events (spikes, incidents)
        random_events = self._generate_random_events(n_days, frequency=0.05)
        
        # 4. Noise
        noise = np.random.normal(0, noise_level, n_days)
        
        # CPU Usage (30-95%)
        cpu_base = 60
        cpu_usage = (
            cpu_base + 
            trend_component * 10 +
            15 * weekly_pattern +
            10 * monthly_pattern +
            random_events * 20 +
            noise * 10
        )
        cpu_usage = np.clip(cpu_usage, 20, 98)
        
        # Memory Usage (40-90%)
        memory_base = 65
        memory_usage = (
            memory_base +
            trend_component * 8 +
            12 * weekly_pattern +
            8 * monthly_pattern +
            random_events * 15 +
            noise * 8
        )
        memory_usage = np.clip(memory_usage, 30, 95)
        
        # Network Traffic (GB)
        network_base = 250
        network_traffic = (
            network_base +
            trend_component * 50 +
            100 * weekly_pattern +
            80 * monthly_pattern +
            random_events * 150 +
            noise * 30
        )
        network_traffic = np.clip(network_traffic, 50, 800)
        
        # Storage Usage (GB) - steadily increasing
        storage_base = 500
        storage_usage = (
            storage_base +
            trend_component * 20 +
            np.cumsum(np.random.uniform(-2, 5, n_days))
        )
        storage_usage = np.clip(storage_usage, 400, 2000)
        
        # Request Count (thousands)
        requests_base = 100
        request_count = (
            requests_base +
            trend_component * 30 +
            50 * weekly_pattern +
            40 * monthly_pattern +
            random_events * 80 +
            noise * 20
        )
        request_count = np.clip(request_count, 20, 500)
        
        # Calculate Cost based on resource usage
        # Pricing model (simplified):
        # - CPU: $0.05 per vCPU-hour
        # - Memory: $0.01 per GB-hour
        # - Network: $0.12 per GB
        # - Storage: $0.023 per GB-month
        # - Requests: $0.20 per million requests
        
        cost = (
            (cpu_usage / 100 * 8 * 24 * 0.05) +  # 8 vCPUs
            (memory_usage / 100 * 32 * 24 * 0.01) +  # 32 GB RAM
            (network_traffic * 0.12) +
            (storage_usage * 0.023 / 30) +  # Daily storage cost
            (request_count * 1000 / 1_000_000 * 0.20)  # Request cost
        )
        
        # Add trend and patterns to cost
        cost = cost * (1 + trend_component) * yearly_pattern
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'date': dates.date,
            'day_of_week': day_of_week,
            'month': month,
            'cost': np.round(cost, 2),
            'cpu_usage': np.round(cpu_usage, 2),
            'memory_usage': np.round(memory_usage, 2),
            'network_traffic': np.round(network_traffic, 2),
            'storage_usage': np.round(storage_usage, 2),
            'request_count': np.round(request_count, 2),
            'is_weekend': (day_of_week >= 5).astype(int),
            'is_event': (random_events > 0).astype(int)
        })
        
        return df
    
    def _generate_random_events(self, n_days: int, frequency: float = 0.05) -> np.ndarray:
        """
        Generate random spike events (traffic surges, incidents)
        
        Args:
            n_days: Number of days
            frequency: Probability of event per day
        
        Returns:
            Array of event intensities
        """
        events = np.zeros(n_days)
        event_indices = np.random.choice(
            n_days,
            size=int(n_days * frequency),
            replace=False
        )
        
        # Random event intensities
        events[event_indices] = np.random.uniform(0.5, 2.0, len(event_indices))
        
        # Events can last multiple days (with decay)
        for idx in event_indices:
            duration = np.random.randint(1, 4)  # 1-3 days
            for d in range(1, duration):
                if idx + d < n_days:
                    events[idx + d] = events[idx] * (0.5 ** d)
        
        return events
    
    def generate_instance_types_data(self) -> pd.DataFrame:
        """
        Generate cloud instance types with specifications and costs
        
        Returns:
            DataFrame with instance type information
        """
        # AWS EC2-like instance types
        instances = [
            # General Purpose
            {'provider': 'aws', 'type': 't3.micro', 'vcpu': 2, 'memory': 1, 
             'cost_per_hour': 0.0104, 'category': 'burstable'},
            {'provider': 'aws', 'type': 't3.small', 'vcpu': 2, 'memory': 2,
             'cost_per_hour': 0.0208, 'category': 'burstable'},
            {'provider': 'aws', 'type': 't3.medium', 'vcpu': 2, 'memory': 4,
             'cost_per_hour': 0.0416, 'category': 'burstable'},
            {'provider': 'aws', 'type': 'm5.large', 'vcpu': 2, 'memory': 8,
             'cost_per_hour': 0.096, 'category': 'general'},
            {'provider': 'aws', 'type': 'm5.xlarge', 'vcpu': 4, 'memory': 16,
             'cost_per_hour': 0.192, 'category': 'general'},
            {'provider': 'aws', 'type': 'm5.2xlarge', 'vcpu': 8, 'memory': 32,
             'cost_per_hour': 0.384, 'category': 'general'},
            {'provider': 'aws', 'type': 'm5.4xlarge', 'vcpu': 16, 'memory': 64,
             'cost_per_hour': 0.768, 'category': 'general'},
            
            # Compute Optimized
            {'provider': 'aws', 'type': 'c5.large', 'vcpu': 2, 'memory': 4,
             'cost_per_hour': 0.085, 'category': 'compute'},
            {'provider': 'aws', 'type': 'c5.xlarge', 'vcpu': 4, 'memory': 8,
             'cost_per_hour': 0.17, 'category': 'compute'},
            {'provider': 'aws', 'type': 'c5.2xlarge', 'vcpu': 8, 'memory': 16,
             'cost_per_hour': 0.34, 'category': 'compute'},
            
            # Memory Optimized
            {'provider': 'aws', 'type': 'r5.large', 'vcpu': 2, 'memory': 16,
             'cost_per_hour': 0.126, 'category': 'memory'},
            {'provider': 'aws', 'type': 'r5.xlarge', 'vcpu': 4, 'memory': 32,
             'cost_per_hour': 0.252, 'category': 'memory'},
            {'provider': 'aws', 'type': 'r5.2xlarge', 'vcpu': 8, 'memory': 64,
             'cost_per_hour': 0.504, 'category': 'memory'},
        ]
        
        df = pd.DataFrame(instances)
        df['cost_per_day'] = df['cost_per_hour'] * 24
        df['cost_per_month'] = df['cost_per_hour'] * 24 * 30
        
        return df
    
    def generate_configuration_scenarios(
        self,
        historical_data: pd.DataFrame
    ) -> list:
        """
        Generate different configuration scenarios for optimization
        
        Args:
            historical_data: Historical usage data
        
        Returns:
            List of configuration dictionaries
        """
        # Analyze current usage patterns
        avg_cpu = historical_data['cpu_usage'].mean()
        max_cpu = historical_data['cpu_usage'].quantile(0.95)
        avg_memory = historical_data['memory_usage'].mean()
        max_memory = historical_data['memory_usage'].quantile(0.95)
        
        scenarios = []
        
        # Scenario 1: Current (baseline)
        scenarios.append({
            'name': 'Current Configuration',
            'instance_type': 'm5.2xlarge',
            'instance_count': 2,
            'vcpu': 8,
            'memory': 32,
            'auto_scaling': False,
            'min_instances': 2,
            'max_instances': 2,
            'expected_cpu': avg_cpu,
            'expected_memory': avg_memory
        })
        
        # Scenario 2: Right-sized (optimal for average)
        scenarios.append({
            'name': 'Right-Sized',
            'instance_type': 'm5.xlarge',
            'instance_count': 2,
            'vcpu': 4,
            'memory': 16,
            'auto_scaling': False,
            'min_instances': 2,
            'max_instances': 2,
            'expected_cpu': avg_cpu * 1.5,  # Will use more % of smaller instance
            'expected_memory': avg_memory * 1.5
        })
        
        # Scenario 3: Auto-scaling enabled
        scenarios.append({
            'name': 'Auto-Scaling Enabled',
            'instance_type': 'm5.xlarge',
            'instance_count': 1,  # Starting count
            'vcpu': 4,
            'memory': 16,
            'auto_scaling': True,
            'min_instances': 1,
            'max_instances': 4,
            'expected_cpu': avg_cpu * 1.3,
            'expected_memory': avg_memory * 1.3
        })
        
        # Scenario 4: Burstable instances (for variable workloads)
        scenarios.append({
            'name': 'Burstable Instances',
            'instance_type': 't3.xlarge',
            'instance_count': 2,
            'vcpu': 4,
            'memory': 16,
            'auto_scaling': False,
            'min_instances': 2,
            'max_instances': 2,
            'expected_cpu': avg_cpu * 1.2,
            'expected_memory': avg_memory * 1.2
        })
        
        # Scenario 5: Compute-optimized (if CPU-heavy workload)
        if avg_cpu > 60:
            scenarios.append({
                'name': 'Compute-Optimized',
                'instance_type': 'c5.2xlarge',
                'instance_count': 2,
                'vcpu': 8,
                'memory': 16,
                'auto_scaling': False,
                'min_instances': 2,
                'max_instances': 2,
                'expected_cpu': avg_cpu * 0.9,  # Better CPU, will use less %
                'expected_memory': avg_memory * 1.8  # Less memory, higher %
            })
        
        # Scenario 6: Reserved instances (1-year commitment)
        scenarios.append({
            'name': 'Reserved Instances (1-year)',
            'instance_type': 'm5.2xlarge',
            'instance_count': 2,
            'vcpu': 8,
            'memory': 32,
            'auto_scaling': False,
            'min_instances': 2,
            'max_instances': 2,
            'reserved': True,
            'discount': 0.40,  # 40% discount
            'expected_cpu': avg_cpu,
            'expected_memory': avg_memory
        })
        
        return scenarios


def save_sample_data():
    """Generate and save sample datasets"""
    generator = CloudDataGenerator(seed=42)
    
    # Generate 2 years of daily data
    print("Generating daily usage data (2023-2024)...")
    daily_data = generator.generate_daily_data(
        start_date="2023-01-01",
        end_date="2024-12-31",
        base_cost=1000.0,
        trend=0.05,
        seasonality=True,
        noise_level=0.15
    )
    
    # Save to CSV
    daily_data.to_csv('/home/claude/cloudcost-optimizer/data/sample_data/daily_usage.csv', index=False)
    print(f"✓ Saved daily usage data: {len(daily_data)} days")
    
    # Generate instance types data
    print("\nGenerating instance types data...")
    instances = generator.generate_instance_types_data()
    instances.to_csv('/home/claude/cloudcost-optimizer/data/sample_data/instance_types.csv', index=False)
    print(f"✓ Saved instance types: {len(instances)} types")
    
    # Generate scenarios
    print("\nGenerating optimization scenarios...")
    scenarios = generator.generate_configuration_scenarios(daily_data)
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df.to_csv('/home/claude/cloudcost-optimizer/data/sample_data/scenarios.csv', index=False)
    print(f"✓ Saved scenarios: {len(scenarios)} configurations")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nDate Range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print(f"Total Days: {len(daily_data)}")
    print(f"\nCost Statistics:")
    print(f"  Average Daily Cost: ${daily_data['cost'].mean():.2f}")
    print(f"  Total Cost: ${daily_data['cost'].sum():,.2f}")
    print(f"  Min/Max Daily: ${daily_data['cost'].min():.2f} / ${daily_data['cost'].max():.2f}")
    print(f"\nResource Usage (Average):")
    print(f"  CPU: {daily_data['cpu_usage'].mean():.1f}%")
    print(f"  Memory: {daily_data['memory_usage'].mean():.1f}%")
    print(f"  Network: {daily_data['network_traffic'].mean():.1f} GB/day")
    print(f"  Storage: {daily_data['storage_usage'].mean():.1f} GB")
    print(f"  Requests: {daily_data['request_count'].mean():.1f}K/day")
    print("\n✓ Sample data generation complete!")


if __name__ == "__main__":
    save_sample_data()
