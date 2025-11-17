#!/usr/bin/env python3
"""
Generate MSA Sequence-Level Model Architecture Diagram

Creates a visual diagram of the model architecture using matplotlib.
Can be saved as PNG or PDF for inclusion in reports/papers.

Usage:
    python generate_architecture_diagram.py
    
Output:
    msa_seqlevel_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_encoder = '#D4E6F1'
color_fusion = '#FAD7A0'
color_pool = '#F8C471'
color_head = '#F5B7B1'
color_output = '#D7BDE2'
color_arrow = '#34495E'

# Helper functions
def draw_box(x, y, width, height, text, color, fontsize=10, style='round'):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"{style},pad=0.05",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight='bold',
        wrap=True
    )

def draw_arrow(x1, y1, x2, y2, style='->'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color_arrow,
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)

def draw_multi_box(x, y, width, height, texts, color, title=None):
    """Draw a box containing multiple sub-components."""
    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor='white',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box)
    
    # Title
    if title:
        ax.text(
            x + width/2, y + height - 0.15,
            title,
            ha='center', va='top',
            fontsize=11,
            fontweight='bold'
        )
    
    # Sub-boxes
    n = len(texts)
    sub_height = (height - 0.4) / n
    for i, text in enumerate(texts):
        sub_y = y + 0.2 + i * sub_height
        sub_box = FancyBboxPatch(
            (x + 0.1, sub_y), width - 0.2, sub_height - 0.05,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(sub_box)
        ax.text(
            x + width/2, sub_y + sub_height/2 - 0.025,
            text,
            ha='center', va='center',
            fontsize=8
        )

# ========== LAYER 1: INPUT ==========
y_current = 18.5
draw_box(0.5, y_current, 2.5, 0.8, 'Text\n(B, 20, 300)', color_input, 9)
draw_box(3.8, y_current, 2.5, 0.8, 'Audio\n(B, 20, 74)', color_input, 9)
draw_box(7.1, y_current, 2.5, 0.8, 'Video\n(B, 20, 47/713)', color_input, 9)

# Title
ax.text(5, 19.5, 'MSA Sequence-Level Model Architecture', 
        ha='center', fontsize=16, fontweight='bold')

# ========== LAYER 2: MODALITY ENCODERS ==========
y_current = 15.5
encoder_height = 2.2

# Text Encoder
draw_multi_box(
    0.5, y_current, 2.5, encoder_height,
    ['Dense(D)', 'Pos. Embed', 'Dropout', 'Transformer\nBlocks × n'],
    color_encoder,
    title='Text Encoder'
)

# Audio Encoder
draw_multi_box(
    3.8, y_current, 2.5, encoder_height,
    ['Dense(D)', 'Pos. Embed', 'Dropout', 'Transformer\nBlocks × n'],
    color_encoder,
    title='Audio Encoder'
)

# Video Encoder
draw_multi_box(
    7.1, y_current, 2.5, encoder_height,
    ['Dense(D)', 'Pos. Embed', 'Dropout', 'Transformer\nBlocks × n'],
    color_encoder,
    title='Video Encoder'
)

# Arrows from input to encoders
draw_arrow(1.75, 18.5, 1.75, 17.7)
draw_arrow(5.05, 18.5, 5.05, 17.7)
draw_arrow(8.35, 18.5, 8.35, 17.7)

# Output dimensions
ax.text(1.75, 15.2, '(B, 20, D)', ha='center', fontsize=8, style='italic')
ax.text(5.05, 15.2, '(B, 20, D)', ha='center', fontsize=8, style='italic')
ax.text(8.35, 15.2, '(B, 20, D)', ha='center', fontsize=8, style='italic')

# ========== LAYER 3: CROSS-MODAL FUSION ==========
y_current = 11.8
fusion_height = 2.8

# Main fusion box
box = FancyBboxPatch(
    (0.5, y_current), 9, fusion_height,
    boxstyle="round,pad=0.05",
    facecolor='white',
    edgecolor='black',
    linewidth=2
)
ax.add_patch(box)

ax.text(5, y_current + fusion_height - 0.2, 'Cross-Modal Fusion',
        ha='center', fontsize=11, fontweight='bold')

# Fusion operations
fusion_ops = [
    'Text ← Audio (Cross-Attention)',
    'Text ← Video (Cross-Attention)',
    '',
    '[Optional Bidirectional]',
    'Audio ← Text (Cross-Attention)',
    'Video ← Text (Cross-Attention)'
]

y_op = y_current + 2.2
for i, op in enumerate(fusion_ops):
    if op:
        fontsize = 8 if '[' not in op else 7
        weight = 'bold' if '[' not in op else 'normal'
        style = 'normal' if '[' not in op else 'italic'
        color_text = 'black' if '[' not in op else 'gray'
        ax.text(5, y_op - i*0.35, op, ha='center', fontsize=fontsize,
                fontweight=weight, style=style, color=color_text)

# Arrows to fusion
draw_arrow(1.75, 15.3, 1.75, 14.6)
draw_arrow(5.05, 15.3, 5.05, 14.6)
draw_arrow(8.35, 15.3, 8.35, 14.6)

# Output dimensions
ax.text(1.75, 11.5, '(B, 20, D)', ha='center', fontsize=8, style='italic')
ax.text(5.05, 11.5, '(B, 20, D)', ha='center', fontsize=8, style='italic')
ax.text(8.35, 11.5, '(B, 20, D)', ha='center', fontsize=8, style='italic')

# ========== LAYER 4: POOLING ==========
y_current = 9.5
pool_height = 1.2

draw_box(0.5, y_current, 2.5, pool_height, 
         'Mask-Aware\nPooling', color_pool, 9)
draw_box(3.8, y_current, 2.5, pool_height, 
         'Mask-Aware\nPooling', color_pool, 9)
draw_box(7.1, y_current, 2.5, pool_height, 
         'Mask-Aware\nPooling', color_pool, 9)

# Arrows to pooling
draw_arrow(1.75, 11.8, 1.75, 10.7)
draw_arrow(5.05, 11.8, 5.05, 10.7)
draw_arrow(8.35, 11.8, 8.35, 10.7)

# Output dimensions
ax.text(1.75, 9.2, '(B, D)', ha='center', fontsize=8, style='italic')
ax.text(5.05, 9.2, '(B, D)', ha='center', fontsize=8, style='italic')
ax.text(8.35, 9.2, '(B, D)', ha='center', fontsize=8, style='italic')

# ========== LAYER 5: ADAPTIVE FUSION HEAD ==========
y_current = 5.5
fusion_head_height = 3.2

# Main box
box = FancyBboxPatch(
    (1.5, y_current), 7, fusion_head_height,
    boxstyle="round,pad=0.05",
    facecolor='white',
    edgecolor='black',
    linewidth=2
)
ax.add_patch(box)

ax.text(5, y_current + fusion_head_height - 0.2, 'Adaptive Fusion Head',
        ha='center', fontsize=11, fontweight='bold')

# Stack modalities
draw_box(2, y_current + 2.2, 6, 0.5, 
         'Stack Modalities → (B, 3, D)', color_head, 8)

# Compute weights
draw_box(2, y_current + 1.4, 6, 0.5,
         'Compute Fusion Weights: Dense(D/2) → Dense(3, softmax)',
         color_head, 7)

# Weighted sum
draw_box(2, y_current + 0.6, 6, 0.5,
         'Weighted Sum → (B, D)', color_head, 8)

# Arrows converging to stack
draw_arrow(1.75, 9.5, 3.5, 8.7)
draw_arrow(5.05, 9.5, 5.0, 8.7)
draw_arrow(8.35, 9.5, 6.5, 8.7)

# Internal arrows
draw_arrow(5, 8.2, 5, 7.9)
draw_arrow(5, 7.4, 5, 7.1)

# Output dimension
ax.text(5, 5.2, '(B, D)', ha='center', fontsize=8, style='italic')

# ========== LAYER 6: REGRESSION HEAD ==========
y_current = 3.0
draw_box(2.5, y_current, 5, 0.8,
         'Regression Head\nDense(1, dtype=float32)',
         color_output, 10)

# Arrow to regression
draw_arrow(5, 5.5, 5, 3.8)

# ========== LAYER 7: OUTPUT ==========
y_current = 1.5
draw_box(3.5, y_current, 3, 0.8,
         'Sentiment Score\n(B, 1)',
         color_output, 10)

# Final arrow
draw_arrow(5, 3.0, 5, 2.3)

# ========== LEGEND ==========
legend_x = 0.3
legend_y = 0.3

ax.text(legend_x, legend_y + 0.6, 'Legend:', fontsize=9, fontweight='bold')
ax.text(legend_x, legend_y + 0.4, 'B = Batch size', fontsize=7)
ax.text(legend_x, legend_y + 0.2, 'D = model_dim', fontsize=7)
ax.text(legend_x, legend_y + 0.0, '20 = Sequence length', fontsize=7)

# Key design features box
feature_x = 5.5
feature_y = 0.1
feature_box = FancyBboxPatch(
    (feature_x, feature_y), 4, 1.2,
    boxstyle="round,pad=0.05",
    facecolor='#F0F0F0',
    edgecolor='black',
    linewidth=1
)
ax.add_patch(feature_box)

ax.text(feature_x + 2, feature_y + 1.0, 'Key Features:', 
        ha='center', fontsize=8, fontweight='bold')
features = [
    '• No early pooling (full sequences)',
    '• Text as anchor modality',
    '• Mask-aware processing',
    '• Learnable modality weights'
]
for i, feat in enumerate(features):
    ax.text(feature_x + 0.15, feature_y + 0.75 - i*0.18, feat,
            fontsize=6, va='top')

# Save figure
plt.tight_layout()
plt.savefig('msa_seqlevel_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('msa_seqlevel_architecture.pdf', bbox_inches='tight')

print("✓ Architecture diagrams saved:")
print("  - msa_seqlevel_architecture.png (high resolution)")
print("  - msa_seqlevel_architecture.pdf (vector format)")
print("\nYou can include these in your report or presentation.")

