from collections import defaultdict


class SpatialHash:
    def __init__(
        self,
        cell_size=0.1
    ):
        self.cell_size = cell_size
        self.hash_table = defaultdict(set)

    def _get_cell_coords(self, x, y):
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return int(x // self.cell_size), int(y // self.cell_size)

    def _get_intersecting_cells(self, bbox):
        cells = set()
        x1, y1, x2, y2 = bbox

        min_x, min_y = self._get_cell_coords(x1, y1)
        max_x, max_y = self._get_cell_coords(x2, y2)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cells.add((x, y))
        return cells

    def insert(self, key, bbox):
        cells = self._get_intersecting_cells(bbox)
        for cell in cells:
            self.hash_table[cell].add(key)

    def query_candidates(self, bbox):
        cells = self._get_intersecting_cells(bbox)
        candidates = set()

        for cell in cells:
            candidates.update(self.hash_table[cell])

        return candidates

    def clear(self):
        self.hash_table.clear()