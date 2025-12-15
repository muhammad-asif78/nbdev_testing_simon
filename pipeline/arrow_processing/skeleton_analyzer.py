import math
import numpy as np
import cv2
from skimage.morphology import skeletonize

def skeleton_has_cycle(skeleton):
    h, w = skeleton.shape
    
    skeleton_pixels = np.count_nonzero(skeleton)
    if skeleton_pixels > 3000: 
        junctions = 0
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] > 0:
                    neighbor_count = np.sum(skeleton[y-1:y+2, x-1:x+2] > 0) - 1
                    if neighbor_count > 2:
                        junctions += 1
                        if junctions > 2:  
                            return True
        return junctions > 0
    
    visited = np.zeros((h, w), dtype=bool)

    def neighbors(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc] > 0:
                    yield nr, nc

    def dfs(r, c, pr, pc, depth=0):
        if depth > 500:  
            return False
        visited[r, c] = True
        for nr, nc in neighbors(r, c):
            if not visited[nr, nc]:
                if dfs(nr, nc, r, c, depth + 1):
                    return True
            elif (nr, nc) != (pr, pc):
                return True
        return False

    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0 and not visited[y, x]:
                if dfs(y, x, -1, -1):
                    return True
    return False

def extract_skeleton(binary_mask):
    return skeletonize(binary_mask > 0).astype(np.uint8) * 255

def get_skeleton_graph_nodes(skeleton):
    endpoints = []
    h, w = skeleton.shape
    padded = np.pad(skeleton, 1, 'constant')
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] > 0:
                n = np.sum(padded[y-1:y+2, x-1:x+2] > 0) - 1
                if n == 1:
                    endpoints.append((x - 1, y - 1))
    return endpoints

def bfs_get_path(skeleton, start_xy, end_xy):
    from collections import deque
    h, w = skeleton.shape
    s, t = (start_xy[1], start_xy[0]), (end_xy[1], end_xy[0])
    if not (0 <= s[0] < h and 0 <= s[1] < w and skeleton[s] > 0):
        return None
    if not (0 <= t[0] < h and 0 <= t[1] < w and skeleton[t] > 0):
        return None
    q = deque([(s, [start_xy])])
    seen = {s}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == t:
            return path
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc] > 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(((nr, nc), path + [(nc, nr)]))
    return None

def find_best_arrow_by_straightness(binary_mask, min_path_length=20):
    skeleton = extract_skeleton(binary_mask)
    if np.count_nonzero(skeleton) < min_path_length:
        return None
    endpoints = get_skeleton_graph_nodes(skeleton)
    if len(endpoints) < 2:
        from ..shape_processing.geometry_utils import find_farthest_points_euclidean
        return find_farthest_points_euclidean(binary_mask)
    import itertools
    candidates = []
    for p1, p2 in itertools.combinations(endpoints, 2):
        path = bfs_get_path(skeleton, p1, p2)
        if path and len(path) >= min_path_length:
            L = len(path)
            D = math.dist(p1, p2)
            score = D / L * L
            candidates.append({"endpoints": [p1, p2], "score": score})
    if not candidates:
        return None
    return max(candidates, key=lambda x: x['score'])['endpoints']
