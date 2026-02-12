"""
Chart components for the AI Fitness Trainer Streamlit application.

This module provides reusable chart components using Plotly for data visualization
with consistent professional styling across the application.
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from styles.theme import COLORS, TYPOGRAPHY


class ChartComponents:
    """
    Reusable chart components with professional styling.
    
    This class provides static methods for creating various types of charts
    (line, bar, pie) with consistent theming and styling.
    """
    
    @staticmethod
    def get_responsive_height(default_height: int = 400) -> int:
        """
        Calculate responsive chart height based on viewport.
        
        Args:
            default_height: Default height in pixels
            
        Returns:
            Adjusted height for responsive display
        """
        # For Streamlit, we'll use a reasonable default
        # In a real implementation, this could use JavaScript to detect viewport
        return default_height
    
    @staticmethod
    def create_line_chart(
        data: Dict[str, List],
        title: str,
        x_label: str,
        y_label: str,
        show_legend: bool = True,
        height: int = 400
    ) -> go.Figure:
        """
        Create a professional line chart for time series data.
        
        Args:
            data: Dictionary with 'x' and 'y' keys containing data points.
                  Can also include 'name' for series label and multiple series.
                  Format: {'x': [...], 'y': [...], 'name': '...'} or
                         [{'x': [...], 'y': [...], 'name': '...'}, ...]
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            show_legend: Whether to display the legend
            height: Chart height in pixels
            
        Returns:
            Plotly Figure object with professional styling applied
        """
        fig = go.Figure()
        
        # Handle single series or multiple series
        if isinstance(data, dict) and 'x' in data and 'y' in data:
            # Single series
            series_name = data.get('name', 'Series')
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines+markers',
                name=series_name,
                line=dict(
                    color=COLORS['primary'],
                    width=3
                ),
                marker=dict(
                    size=8,
                    color=COLORS['primary'],
                    line=dict(color=COLORS['background'], width=2)
                ),
                hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
            ))
        elif isinstance(data, list):
            # Multiple series
            colors = [
                COLORS['primary'],
                COLORS['secondary'],
                COLORS['accent'],
                COLORS['warning'],
                COLORS['info']
            ]
            for idx, series in enumerate(data):
                color = colors[idx % len(colors)]
                series_name = series.get('name', f'Series {idx + 1}')
                fig.add_trace(go.Scatter(
                    x=series['x'],
                    y=series['y'],
                    mode='lines+markers',
                    name=series_name,
                    line=dict(
                        color=color,
                        width=3
                    ),
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(color=COLORS['background'], width=2)
                    ),
                    hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
                ))
        
        # Apply professional theme
        fig = ChartComponents.apply_professional_theme(
            fig, title, x_label, y_label, show_legend, height
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(
        data: Dict[str, List],
        title: str,
        x_label: str,
        y_label: str,
        orientation: str = 'v',
        show_legend: bool = False,
        height: int = 400
    ) -> go.Figure:
        """
        Create a professional bar chart for categorical data.
        
        Args:
            data: Dictionary with 'categories' and 'values' keys.
                  Format: {'categories': [...], 'values': [...]} or
                         [{'categories': [...], 'values': [...], 'name': '...'}, ...]
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            orientation: 'v' for vertical bars, 'h' for horizontal bars
            show_legend: Whether to display the legend
            height: Chart height in pixels
            
        Returns:
            Plotly Figure object with professional styling applied
        """
        fig = go.Figure()
        
        # Handle single series or multiple series
        if isinstance(data, dict) and 'categories' in data and 'values' in data:
            # Single series
            series_name = data.get('name', 'Values')
            if orientation == 'v':
                fig.add_trace(go.Bar(
                    x=data['categories'],
                    y=data['values'],
                    name=series_name,
                    marker=dict(
                        color=COLORS['primary'],
                        line=dict(color=COLORS['primary_dark'], width=1)
                    ),
                    hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Bar(
                    x=data['values'],
                    y=data['categories'],
                    name=series_name,
                    orientation='h',
                    marker=dict(
                        color=COLORS['primary'],
                        line=dict(color=COLORS['primary_dark'], width=1)
                    ),
                    hovertemplate='<b>%{y}</b><br>%{x}<extra></extra>'
                ))
        elif isinstance(data, list):
            # Multiple series
            colors = [
                COLORS['primary'],
                COLORS['secondary'],
                COLORS['accent'],
                COLORS['warning'],
                COLORS['info']
            ]
            for idx, series in enumerate(data):
                color = colors[idx % len(colors)]
                series_name = series.get('name', f'Series {idx + 1}')
                if orientation == 'v':
                    fig.add_trace(go.Bar(
                        x=series['categories'],
                        y=series['values'],
                        name=series_name,
                        marker=dict(
                            color=color,
                            line=dict(color=COLORS['border_dark'], width=1)
                        ),
                        hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=series['values'],
                        y=series['categories'],
                        name=series_name,
                        orientation='h',
                        marker=dict(
                            color=color,
                            line=dict(color=COLORS['border_dark'], width=1)
                        ),
                        hovertemplate='<b>%{y}</b><br>%{x}<extra></extra>'
                    ))
        
        # Apply professional theme
        fig = ChartComponents.apply_professional_theme(
            fig, title, x_label, y_label, show_legend, height
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(
        data: Dict[str, List],
        title: str,
        show_legend: bool = True,
        height: int = 400
    ) -> go.Figure:
        """
        Create a professional pie chart for distribution data.
        
        Args:
            data: Dictionary with 'labels' and 'values' keys.
                  Format: {'labels': [...], 'values': [...]}
            title: Chart title
            show_legend: Whether to display the legend
            height: Chart height in pixels
            
        Returns:
            Plotly Figure object with professional styling applied
        """
        fig = go.Figure()
        
        # Create color palette for pie slices
        colors = [
            COLORS['primary'],
            COLORS['secondary'],
            COLORS['accent'],
            COLORS['warning'],
            COLORS['info'],
            COLORS['primary_light'],
            COLORS['secondary_light'],
            COLORS['accent_light']
        ]
        
        # Ensure we have enough colors for all slices
        num_slices = len(data['labels'])
        slice_colors = (colors * ((num_slices // len(colors)) + 1))[:num_slices]
        
        fig.add_trace(go.Pie(
            labels=data['labels'],
            values=data['values'],
            marker=dict(
                colors=slice_colors,
                line=dict(color=COLORS['background'], width=2)
            ),
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>',
            hole=0.3  # Create a donut chart for modern look
        ))
        
        # Apply professional theme (simplified for pie charts)
        fig = ChartComponents.apply_professional_theme(
            fig, title, None, None, show_legend, height
        )
        
        return fig
    
    @staticmethod
    def apply_professional_theme(
        fig: go.Figure,
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        show_legend: bool = True,
        height: int = 400
    ) -> go.Figure:
        """
        Apply consistent professional styling to a Plotly chart.
        
        Args:
            fig: Plotly Figure object to style
            title: Chart title
            x_label: X-axis label (optional, not used for pie charts)
            y_label: Y-axis label (optional, not used for pie charts)
            show_legend: Whether to display the legend
            height: Chart height in pixels
            
        Returns:
            Styled Plotly Figure object
        """
        # Update layout with professional styling and responsive design
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=20,
                    color=COLORS['text_primary'],
                    family=TYPOGRAPHY['font_family_primary']
                ),
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top'
            ),
            font=dict(
                family=TYPOGRAPHY['font_family_primary'],
                size=14,
                color=COLORS['text_primary']
            ),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            height=height,
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5,
                font=dict(
                    size=12,
                    color=COLORS['text_secondary']
                ),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=COLORS['border'],
                borderwidth=1
            ),
            # Responsive configuration
            autosize=True,
            # Ensure chart scales to container width
            width=None,
        )
        
        # Configure responsive behavior
        fig.update_layout(
            modebar=dict(
                bgcolor='rgba(255, 255, 255, 0.8)',
                color=COLORS['text_secondary'],
                activecolor=COLORS['primary']
            )
        )
        
        # Update axes styling (if applicable)
        if x_label is not None:
            fig.update_xaxes(
                title=dict(
                    text=x_label,
                    font=dict(
                        size=14,
                        color=COLORS['text_secondary']
                    )
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS['border'],
                showline=True,
                linewidth=1,
                linecolor=COLORS['border_dark'],
                tickfont=dict(
                    size=12,
                    color=COLORS['text_secondary']
                )
            )
        
        if y_label is not None:
            fig.update_yaxes(
                title=dict(
                    text=y_label,
                    font=dict(
                        size=14,
                        color=COLORS['text_secondary']
                    )
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS['border'],
                showline=True,
                linewidth=1,
                linecolor=COLORS['border_dark'],
                tickfont=dict(
                    size=12,
                    color=COLORS['text_secondary']
                )
            )
        
        return fig
