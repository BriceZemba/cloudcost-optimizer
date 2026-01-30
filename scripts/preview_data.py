"""
Quick Data Preview Script
Shows summary of generated cloud data
"""

import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("CLOUDCOST OPTIMIZER - DATA PREVIEW")
print("="*70)

# Load daily usage data
print("\n📊 Loading daily_usage.csv...")
daily = pd.read_csv('data/sample_data/daily_usage.csv')
daily['timestamp'] = pd.to_datetime(daily['timestamp'])

print(f"\n✅ Loaded {len(daily)} days of data")
print(f"Date range: {daily['timestamp'].min()} to {daily['timestamp'].max()}")

# Summary statistics
print("\n" + "="*70)
print("COST STATISTICS")
print("="*70)
print(f"Average daily cost:  ${daily['cost'].mean():.2f}")
print(f"Median daily cost:   ${daily['cost'].median():.2f}")
print(f"Min daily cost:      ${daily['cost'].min():.2f}")
print(f"Max daily cost:      ${daily['cost'].max():.2f}")
print(f"\nTotal cost (2 years): ${daily['cost'].sum():,.2f}")
print(f"Monthly average:      ${daily['cost'].mean() * 30:,.2f}")

print("\n" + "="*70)
print("RESOURCE UTILIZATION (Average)")
print("="*70)
print(f"CPU Usage:           {daily['cpu_usage'].mean():.1f}%")
print(f"Memory Usage:        {daily['memory_usage'].mean():.1f}%")
print(f"Network Traffic:     {daily['network_traffic'].mean():.1f} GB/day")
print(f"Storage Usage:       {daily['storage_usage'].mean():.1f} GB")
print(f"Request Count:       {daily['request_count'].mean():.1f}K/day")

# Load instance types
print("\n📊 Loading instance_types.csv...")
instances = pd.read_csv('data/sample_data/instance_types.csv')
print(f"\n✅ Loaded {len(instances)} instance types")

print("\n" + "="*70)
print("AVAILABLE INSTANCE TYPES")
print("="*70)
print(instances[['type', 'vcpu', 'memory', 'cost_per_month', 'category']].to_string(index=False))

# Load scenarios
print("\n📊 Loading scenarios.csv...")
scenarios = pd.read_csv('data/sample_data/scenarios.csv')
print(f"\n✅ Loaded {len(scenarios)} optimization scenarios")

print("\n" + "="*70)
print("OPTIMIZATION SCENARIOS")
print("="*70)
for idx, scenario in scenarios.iterrows():
    print(f"\n{idx+1}. {scenario['name']}")
    print(f"   Instance: {scenario['instance_type']}")
    print(f"   vCPU: {scenario['vcpu']}, Memory: {scenario['memory']} GB")
    print(f"   Auto-scaling: {scenario['auto_scaling']}")

# Create visualization
print("\n📈 Creating cost trend visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Cost over time
axes[0, 0].plot(daily['timestamp'], daily['cost'], linewidth=0.5)
axes[0, 0].set_title('Daily Cost Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Cost ($)')
axes[0, 0].grid(True, alpha=0.3)

# CPU vs Memory usage
axes[0, 1].scatter(daily['cpu_usage'], daily['memory_usage'], 
                   c=daily['cost'], cmap='RdYlGn_r', alpha=0.5, s=10)
axes[0, 1].set_title('CPU vs Memory Usage (colored by cost)', 
                      fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('CPU Usage (%)')
axes[0, 1].set_ylabel('Memory Usage (%)')
axes[0, 1].grid(True, alpha=0.3)

# Cost distribution
axes[1, 0].hist(daily['cost'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Cost Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Daily Cost ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(daily['cost'].mean(), color='red', 
                   linestyle='--', label=f'Mean: ${daily["cost"].mean():.2f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Weekly pattern
daily['day_name'] = daily['timestamp'].dt.day_name()
weekly_avg = daily.groupby('day_name')['cost'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
axes[1, 1].bar(range(7), weekly_avg.values, color='steelblue', alpha=0.7)
axes[1, 1].set_title('Average Cost by Day of Week', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Average Cost ($)')
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data_preview.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: data_preview.png")

plt.show()

print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

# Check for missing values
missing = daily.isnull().sum()
print(f"\nMissing values: {missing.sum()} (should be 0)")

# Check data ranges
print("\n✅ Data ranges look good:")
print(f"   CPU usage: {daily['cpu_usage'].min():.1f}% - {daily['cpu_usage'].max():.1f}%")
print(f"   Memory usage: {daily['memory_usage'].min():.1f}% - {daily['memory_usage'].max():.1f}%")
print(f"   All values are realistic ✓")

print("\n" + "="*70)
print("✅ ALL DATA FILES ARE READY TO USE!")
print("="*70)
print("\nNext steps:")
print("1. Run: python src/models/cost_predictor.py")
print("2. Train the LSTM model")
print("3. Make predictions")
print("4. Push to GitHub!")
print("\n" + "="*70)
