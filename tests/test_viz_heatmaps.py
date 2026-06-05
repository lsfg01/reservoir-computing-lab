from __future__ import annotations

import numpy as np


def test_heatmap_annotations_use_rendered_grid_orientation() -> None:
    import matplotlib.pyplot as plt

    from rc_lab.viz.campaign import _annotate_heatmap
    from rc_lab.viz.primitives import heatmap_display_grid, heatmap_plane

    grid = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])
    displayed = heatmap_display_grid(grid)

    fig, ax = plt.subplots()
    try:
        pcm = heatmap_plane(
            ax,
            grid,
            x_ticks=[0.5, 1.0],
            y_ticks=[1.0, 0.1],
            cmap="viridis",
            vmin=float(np.nanmin(grid)),
            vmax=float(np.nanmax(grid)),
        )
        _annotate_heatmap(
            ax,
            displayed,
            vmin=float(np.nanmin(grid)),
            vmax=float(np.nanmax(grid)),
            fmt="{:.0f}",
        )

        colored = np.asarray(pcm.get_array()).reshape(displayed.shape)
        assert np.array_equal(colored, displayed)

        labels_by_position = {
            tuple(int(coord) for coord in text.get_position()): text.get_text()
            for text in ax.texts
        }
        assert labels_by_position == {
            (0, 0): "30",
            (1, 0): "40",
            (0, 1): "10",
            (1, 1): "20",
        }
    finally:
        plt.close(fig)
