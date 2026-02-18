# Jupyter Notebook: Bubble chart from multiple metadata JSON files

import json
import pandas as pd
import matplotlib.pyplot as plt

# === User configuration ===
# List your metadata JSON files here
metadata_files = [
    '',
    '',
    ''  # add more as needed
]
# Frames per second of the source video
fps = 30
# Scale factor for bubble sizes (frames)
size_scale = 1
# Marker diameter (in points) for legend entries
legend_marker_size = 10
# Output path for saved figure
output_path = 'bubble_chart.png'

# Load and combine metadata
data = []
for file_path in metadata_files:
    with open(file_path, 'r') as f:
        try:
            meta = json.load(f)
            data.extend(meta)
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")

# Create DataFrame
df = pd.DataFrame(data)
if df.empty:
    raise ValueError("No metadata loaded. Check your metadata_files list.")

# Aggregate total duration and count per event
grouped = df.groupby('event').agg(
    total_duration=('duration_seconds', 'sum'),
    instances=('clip_filename', 'count')
)
# Compute total frames per event
grouped['total_frames'] = grouped['total_duration'] * fps

# Prepare for plotting
events = grouped.index.tolist()
frames = grouped['total_frames'].values
instances = grouped['instances'].values
x_positions = list(range(len(events)))

# Colormap for distinct classes
cmap = plt.cm.get_cmap('tab10', len(events))

# Plot bubble chart
fig, ax = plt.subplots(figsize=(10, 6))
for i, event in enumerate(events):
    size = frames[i] * size_scale
    ax.scatter(
        x_positions[i],
        instances[i],
        s=size,
        color=cmap(i),
        alpha=0.6
    )

# Labels and formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(events, rotation=45, ha='right')
ax.set_xlabel('Event Class')
ax.set_ylabel('Number of Unique Instances')
ax.set_title('Instances per Event Class\nBubble size ∝ Total Frames')

# Create uniform-size legend handles with fixed marker size
gen_handles = []
for i, event in enumerate(events):
    handle = plt.Line2D(
        [], [],
        marker='o', linestyle='',
        markerfacecolor=cmap(i),
        markersize=legend_marker_size,
        alpha=0.6
    )
    gen_handles.append(handle)

# Legend for event classes with uniform marker sizes
ax.legend(
    gen_handles, events,
    title='Event Class',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.
)

plt.tight_layout()
# Save the figure instead of showing it
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Bubble chart saved to '{output_path}'")
