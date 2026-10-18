import cv2
import numpy as np

THRESHOLD_VALUE = 180
CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 50

ROI_MASK_FACTOR = 0.5
PROCESSING_WIDTH = 1280

MIN_DOODLE_POINTS = 2
MIN_DOODLE_LENGTH = 100

CONNECTION_MAX_ANGLE = 35
CONNECTION_MAX_DISTANCE = 250
CONNECTION_MAX_CONTINUATION_ANGLE = 45
MAX_CONTINUITY_OFFSET = 35
MAX_LOWER_POINT_DISTANCE = 150

STRUCTURE_CONNECTION_THRESHOLD = 0.5
MIN_STRUCTURE_HEIGHT = 100
STRUCTURE_MERGE_THRESHOLD = 0.35
MAX_STRUCTURE_MERGE_GAP = 150
MAX_STRUCTURE_MERGE_SEPARATION = 100
MAX_STRUCTURE_WIDTH = 180

MAX_CONVERGENCE_ORIGIN_DISTANCE = 180
EDGE_TOUCH_MARGIN = 10
LOOKAHEAD_RATIO = 0.75

COLOR_GRID_SIZE = 12
ROAD_COLOR_DISTANCE = 25
ROAD_MIN_REGION_CELLS = 100
ROAD_MIN_BOTTOM_RATIO = 0.80
ROAD_REFERENCE_WIDTH = 0.20
ROAD_REFERENCE_HEIGHT = 0.10


def resize_image(image):
    height, width = image.shape[:2]

    if width <= PROCESSING_WIDTH:
        return image

    scale = PROCESSING_WIDTH / width
    new_height = int(height * scale)

    return cv2.resize(image, (PROCESSING_WIDTH, new_height))


def create_roi(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    points = np.array([
        [(0, height), (width, height), (width, int(height * ROI_MASK_FACTOR)), (0, int(height * ROI_MASK_FACTOR))]
    ], dtype=np.int32)

    cv2.fillPoly(mask, points, (255, 255, 255))

    return cv2.bitwise_and(image, mask)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, CANNY_LOW, CANNY_HIGH)

    return edges


def get_hough_lines(edges):
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        return []

    # Normalize OpenCV's output to an array of (x1, y1, x2, y2).
    lines = np.asarray(lines).reshape(-1, 4)

    return [tuple(map(int, line)) for line in lines]


def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def line_x_at_y(line, y):
    x1, y1, x2, y2 = line

    if y1 == y2:
        return None

    t = (y - y1) / (y2 - y1)

    if 0 <= t <= 1:
        return x1 + t * (x2 - x1)

    return None


def line_angle(line):
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def line_endpoints(line):
    x1, y1, x2, y2 = line
    return [(x1, y1), (x2, y2)]


def lower_point(line):
    x1, y1, x2, y2 = line
    return (x1, y1) if y1 >= y2 else (x2, y2)


def point_line_distance(point, line):
    p = np.array(point, dtype=float)
    a = np.array(line[0], dtype=float)
    b = np.array(line[1], dtype=float)

    direction = b - a
    length = np.linalg.norm(direction)

    if length == 0:
        return float("inf")

    vector = p - a
    cross = direction[0] * vector[1] - direction[1] * vector[0]

    return abs(cross) / length


def lines_are_continuous(current_line, candidate):
    current_points = line_endpoints(current_line)
    candidate_points = line_endpoints(candidate)

    best = None

    for current_connect in current_points:
        current_far = current_points[1] if current_connect == current_points[0] else current_points[0]

        for candidate_connect in candidate_points:
            candidate_far = candidate_points[1] if candidate_connect == candidate_points[0] else candidate_points[0]

            connection = np.array(candidate_connect, dtype=float) - np.array(current_connect, dtype=float)
            connection_length = np.linalg.norm(connection)

            if connection_length == 0:
                continue

            current_vector = np.array(current_connect, dtype=float) - np.array(current_far, dtype=float)
            candidate_vector = np.array(candidate_far, dtype=float) - np.array(candidate_connect, dtype=float)

            current_length = np.linalg.norm(current_vector)
            candidate_length = np.linalg.norm(candidate_vector)

            if current_length == 0 or candidate_length == 0:
                continue

            current_vector /= current_length
            candidate_vector /= candidate_length
            connection_direction = connection / connection_length

            current_alignment = np.dot(current_vector, connection_direction)
            candidate_alignment = np.dot(candidate_vector, -connection_direction)

            if current_alignment <= 0 or candidate_alignment <= 0:
                continue

            offset = point_line_distance(candidate_connect, (current_far, current_connect))
            score = current_alignment + candidate_alignment

            if best is None or score > best[0]:
                best = (score, offset)

    if best is None:
        return False

    return best[1] <= MAX_CONTINUITY_OFFSET


def group_continuous_lines(lines, max_gap=100, max_angle_difference=25):
    if not lines:
        return []

    unused = set(range(len(lines)))
    groups = []

    while unused:
        start = unused.pop()
        group = [lines[start]]
        current_line = lines[start]

        changed = True
        while changed:
            changed = False
            best_index = None
            best_distance = float("inf")
            current_orientation = line_angle(current_line) % 180

            for index in unused:
                candidate = lines[index]
                candidate_orientation = line_angle(candidate) % 180
                orientation_difference = abs(candidate_orientation - current_orientation)

                if orientation_difference > 90:
                    orientation_difference = 180 - orientation_difference

                if orientation_difference > max_angle_difference:
                    continue

                endpoint_distance = min(
                    distance(p1, p2)
                    for p1 in line_endpoints(current_line)
                    for p2 in line_endpoints(candidate)
                )

                if endpoint_distance > max_gap:
                    continue

                if distance(lower_point(current_line), lower_point(candidate)) > MAX_LOWER_POINT_DISTANCE:
                    continue

                if not lines_are_continuous(current_line, candidate):
                    continue

                if endpoint_distance < best_distance:
                    best_distance = endpoint_distance
                    best_index = index

            if best_index is not None:
                current_line = lines[best_index]
                group.append(current_line)
                unused.remove(best_index)
                changed = True

        groups.append(group)

    return groups


def build_group_average_line(group, y_step=40):
    if not group:
        return []

    y_min = min(min(line[1], line[3]) for line in group)
    y_max = max(max(line[1], line[3]) for line in group)

    average_points = []
    y_values = list(range(y_min, y_max + 1, y_step))

    if y_values[-1] != y_max:
        y_values.append(y_max)

    for y in y_values:
        x_values = []

        for line in group:
            x = line_x_at_y(line, y)

            if x is not None:
                x_values.append(x)

        if x_values:
            average_points.append((y, np.mean(x_values)))

    return average_points


def doodle_point_xy(point):
    y, x = point
    return np.array([x, y], dtype=float)


def point_distance(p1, p2):
    return np.linalg.norm(doodle_point_xy(p1) - doodle_point_xy(p2))


def polyline_length(doodle):
    if len(doodle) < 2:
        return 0

    return sum(point_distance(doodle[i], doodle[i + 1]) for i in range(len(doodle) - 1))


def build_doodles(groups):
    doodles = []

    for group in groups:
        doodle = build_group_average_line(group)

        if len(doodle) >= MIN_DOODLE_POINTS and polyline_length(doodle) >= MIN_DOODLE_LENGTH:
            doodles.append(doodle)

    return doodles


def doodle_y_range(doodle):
    ys = [point[0] for point in doodle]
    return min(ys), max(ys)


def doodle_x_at_y(doodle, y):
    if len(doodle) < 2:
        return None

    for i in range(len(doodle) - 1):
        y1, x1 = doodle[i]
        y2, x2 = doodle[i + 1]

        if y1 == y2:
            continue

        if min(y1, y2) <= y <= max(y1, y2):
            t = (y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)

    return None


def doodle_direction(doodle, from_start=True, points=4):
    if len(doodle) < 2:
        return None

    count = min(points, len(doodle) - 1)

    if from_start:
        p1 = doodle[0]
        p2 = doodle[count]
    else:
        p1 = doodle[-1]
        p2 = doodle[-1 - count]

    direction = doodle_point_xy(p2) - doodle_point_xy(p1)
    length = np.linalg.norm(direction)

    if length == 0:
        return None

    return direction / length


def direction_angle(direction):
    if direction is None:
        return None

    return np.degrees(np.arctan2(direction[1], direction[0]))


def doodle_orientation(doodle, points=4):
    direction = doodle_direction(doodle, True, points)

    if direction is None:
        return None

    return direction_angle(direction) % 180


def orientation_difference(angle1, angle2):
    if angle1 is None or angle2 is None:
        return 180

    difference = abs(angle1 - angle2)

    if difference > 90:
        difference = 180 - difference

    return difference


def doodle_position_difference(doodle1, doodle2, samples=5):
    y1_min, y1_max = doodle_y_range(doodle1)
    y2_min, y2_max = doodle_y_range(doodle2)

    y_min = max(y1_min, y2_min)
    y_max = min(y1_max, y2_max)

    if y_min >= y_max:
        return None

    differences = []

    for y in np.linspace(y_min, y_max, samples):
        x1 = doodle_x_at_y(doodle1, y)
        x2 = doodle_x_at_y(doodle2, y)

        if x1 is not None and x2 is not None:
            differences.append(abs(x1 - x2))

    if not differences:
        return None

    return np.mean(differences)


def doodle_continuation_score(doodle1, doodle2):
    endpoints1 = [doodle1[0], doodle1[-1]]
    endpoints2 = [doodle2[0], doodle2[-1]]
    best_score = 0

    for point1 in endpoints1:
        for point2 in endpoints2:
            connection = doodle_point_xy(point2) - doodle_point_xy(point1)
            connection_length = np.linalg.norm(connection)

            if connection_length == 0:
                continue

            connection /= connection_length

            direction1 = doodle_direction(doodle1, point1 == doodle1[0])
            direction2 = doodle_direction(doodle2, point2 == doodle2[0])

            if direction1 is None or direction2 is None:
                continue

            angle1 = np.degrees(np.arccos(np.clip(abs(np.dot(direction1, connection)), -1, 1)))
            angle2 = np.degrees(np.arccos(np.clip(abs(np.dot(direction2, connection)), -1, 1)))
            angle = max(angle1, angle2)

            if angle > CONNECTION_MAX_CONTINUATION_ANGLE:
                continue

            angle_score = 1 - angle / CONNECTION_MAX_CONTINUATION_ANGLE
            distance_score = max(0, 1 - connection_length / CONNECTION_MAX_DISTANCE)
            score = 0.7 * angle_score + 0.3 * distance_score
            best_score = max(best_score, score)

    return best_score


def doodle_connection_score(doodle1, doodle2):
    if len(doodle1) < 2 or len(doodle2) < 2:
        return 0

    orientation1 = doodle_orientation(doodle1)
    orientation2 = doodle_orientation(doodle2)

    if orientation1 is None or orientation2 is None:
        return 0

    angle_difference = orientation_difference(orientation1, orientation2)

    if angle_difference > CONNECTION_MAX_ANGLE:
        return 0

    angle_score = 1 - angle_difference / CONNECTION_MAX_ANGLE

    y1_min, y1_max = doodle_y_range(doodle1)
    y2_min, y2_max = doodle_y_range(doodle2)
    overlap_min = max(y1_min, y2_min)
    overlap_max = min(y1_max, y2_max)

    if overlap_min < overlap_max:
        position_difference = doodle_position_difference(doodle1, doodle2)

        if position_difference is None:
            return 0

        spatial_score = max(0, 1 - position_difference / CONNECTION_MAX_DISTANCE)
    else:
        distances = [
            point_distance(point1, point2)
            for point1 in [doodle1[0], doodle1[-1]]
            for point2 in [doodle2[0], doodle2[-1]]
        ]

        minimum_distance = min(distances)

        if minimum_distance > CONNECTION_MAX_DISTANCE:
            return 0

        spatial_score = 1 - minimum_distance / CONNECTION_MAX_DISTANCE

    continuation_score = doodle_continuation_score(doodle1, doodle2)

    if continuation_score == 0:
        return 0

    return 0.3 * angle_score + 0.3 * spatial_score + 0.4 * continuation_score


def build_structure_groups(doodles):
    structures = []
    used = set()

    for i in range(len(doodles)):
        if i in used:
            continue

        structure = {i}
        changed = True

        while changed:
            changed = False

            for j in range(len(doodles)):
                if j in structure:
                    continue

                for index in list(structure):
                    if doodle_connection_score(doodles[index], doodles[j]) >= STRUCTURE_CONNECTION_THRESHOLD:
                        structure.add(j)
                        changed = True
                        break

        used.update(structure)
        structures.append(sorted(structure))

    return structures


def structure_y_range(structure, doodles):
    ranges = [doodle_y_range(doodles[index]) for index in structure]
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def structure_x_at_y(structure, doodles, y):
    x_values = []

    for index in structure:
        x = doodle_x_at_y(doodles[index], y)

        if x is not None:
            x_values.append(x)

    if not x_values:
        return None

    return np.mean(x_values)


def structure_x_range(structure, doodles):
    y_min, y_max = structure_y_range(structure, doodles)

    return (
        structure_x_at_y(structure, doodles, y_min),
        structure_x_at_y(structure, doodles, y_max)
    )


def is_plausible_edge_pair(structure_a, structure_b, doodles):
    a_y_min, a_y_max = structure_y_range(structure_a, doodles)
    b_y_min, b_y_max = structure_y_range(structure_b, doodles)

    shared_y_min = max(a_y_min, b_y_min)
    shared_y_max = min(a_y_max, b_y_max)

    if shared_y_max - shared_y_min < 40:
        return False

    a_x = structure_x_at_y(structure_a, doodles, shared_y_max)
    b_x = structure_x_at_y(structure_b, doodles, shared_y_max)

    if a_x is None or b_x is None:
        return False

    return abs(b_x - a_x) >= 100


def filter_weak_structures(structures, doodles):
    strong_structures = []
    weak_structures = []

    for structure in structures:
        y_min, y_max = structure_y_range(structure, doodles)

        if y_max - y_min >= MIN_STRUCTURE_HEIGHT:
            strong_structures.append(structure)
        else:
            weak_structures.append(structure)

    for structure in weak_structures:
        if any(is_plausible_edge_pair(structure, other, doodles) for other in strong_structures):
            strong_structures.append(structure)

    if len(strong_structures) < 2:
        for i in range(len(weak_structures)):
            for j in range(i + 1, len(weak_structures)):
                if is_plausible_edge_pair(weak_structures[i], weak_structures[j], doodles):
                    strong_structures.extend([weak_structures[i], weak_structures[j]])
                    return strong_structures

    return strong_structures


def structure_vertical_relationship(structure1, structure2, doodles):
    y1_min, y1_max = structure_y_range(structure1, doodles)
    y2_min, y2_max = structure_y_range(structure2, doodles)

    overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    gap = max(y1_min - y2_max, y2_min - y1_max, 0)

    return overlap, gap


def structure_horizontal_separation(structure1, structure2, doodles, samples=5):
    y1_min, y1_max = structure_y_range(structure1, doodles)
    y2_min, y2_max = structure_y_range(structure2, doodles)

    y_min = max(y1_min, y2_min)
    y_max = min(y1_max, y2_max)

    if y_min >= y_max:
        return None

    separations = []

    for y in np.linspace(y_min, y_max, samples):
        x1 = structure_x_at_y(structure1, doodles, y)
        x2 = structure_x_at_y(structure2, doodles, y)

        if x1 is not None and x2 is not None:
            separations.append(abs(x1 - x2))

    if not separations:
        return None

    return np.mean(separations)


def structure_width(structure, doodles):
    x_min, x_max = structure_x_range(structure, doodles)
    return x_max - x_min


def should_merge_structures(structure1, structure2, doodles):
    best_score = 0

    for doodle_i in structure1:
        for doodle_j in structure2:
            best_score = max(best_score, doodle_connection_score(doodles[doodle_i], doodles[doodle_j]))

    overlap, gap = structure_vertical_relationship(structure1, structure2, doodles)
    separation = structure_horizontal_separation(structure1, structure2, doodles)
    width = structure_width(structure1 + structure2, doodles)

    if best_score < STRUCTURE_MERGE_THRESHOLD:
        return False

    if overlap > 0 or gap > MAX_STRUCTURE_MERGE_GAP:
        return False

    if separation is not None and separation > MAX_STRUCTURE_MERGE_SEPARATION:
        return False

    return width <= MAX_STRUCTURE_WIDTH


def merge_structures(structures, doodles):
    merged = []
    used = set()

    for i in range(len(structures)):
        if i in used:
            continue

        current = structures[i]
        merged_with = None

        for j in range(i + 1, len(structures)):
            if j in used:
                continue

            if should_merge_structures(current, structures[j], doodles):
                merged_with = j
                break

        if merged_with is not None:
            current = current + structures[merged_with]
            used.add(merged_with)

        merged.append(current)
        used.add(i)

    return merged


def is_image_border_structure(structure, doodles, image_width):
    _, bottom_x = structure_x_range(structure, doodles)
    return bottom_x <= EDGE_TOUCH_MARGIN or bottom_x >= image_width - 1 - EDGE_TOUCH_MARGIN


def generate_structure_candidates(structures, doodles):
    candidates = []

    for left in range(len(structures)):
        for middle in range(len(structures)):
            for right in range(len(structures)):
                if len({left, middle, right}) < 3:
                    continue

                _, left_bottom = structure_x_range(structures[left], doodles)
                _, middle_bottom = structure_x_range(structures[middle], doodles)
                _, right_bottom = structure_x_range(structures[right], doodles)

                if left_bottom < middle_bottom < right_bottom:
                    candidates.append((left, middle, right))

    return candidates


def candidate_shared_height(left, middle, right, doodles):
    y_ranges = [
        structure_y_range(structure, doodles)
        for structure in (left, middle, right)
    ]

    y_min = max(y_range[0] for y_range in y_ranges)
    y_max = min(y_range[1] for y_range in y_ranges)

    return max(0, y_max - y_min)


def filter_candidates_by_shared_height(candidates, structures, doodles):
    heights = []

    for candidate in candidates:
        left, middle, right = candidate
        heights.append(candidate_shared_height(structures[left], structures[middle], structures[right], doodles))

    if not heights:
        return []

    minimum_height = 0.5 * max(heights)

    return [
        candidate
        for candidate, height in zip(candidates, heights)
        if height >= minimum_height
    ]


def evaluate_candidate(left, middle, right, doodles, samples=5):
    structures = [left, middle, right]
    y_ranges = [structure_y_range(structure, doodles) for structure in structures]

    y_min = max(y_range[0] for y_range in y_ranges)
    y_max = min(y_range[1] for y_range in y_ranges)

    if y_min >= y_max:
        return None

    spacing_errors = []

    for y in np.linspace(y_min, y_max, samples):
        x_values = [structure_x_at_y(structure, doodles, y) for structure in structures]

        if None in x_values:
            continue

        left_x, middle_x, right_x = x_values
        left_gap = middle_x - left_x
        right_gap = right_x - middle_x

        if left_gap <= 0 or right_gap <= 0:
            continue

        spacing_errors.append(abs(left_gap - right_gap) / max(left_gap, right_gap))

    if not spacing_errors:
        return None

    return np.sqrt(np.mean(np.square(spacing_errors)))


def preliminary_candidate_is_valid(structures, doodles):
    if len(structures) != 3:
        return False

    left, middle, right = structures

    _, left_bottom = structure_x_range(left, doodles)
    _, middle_bottom = structure_x_range(middle, doodles)
    _, right_bottom = structure_x_range(right, doodles)

    if not left_bottom < middle_bottom < right_bottom:
        return False

    if candidate_shared_height(left, middle, right, doodles) <= 0:
        return False

    return evaluate_candidate(left, middle, right, doodles) is not None


def sort_structures_left_to_right(structures, doodles):
    return sorted(structures, key=lambda structure: structure_x_range(structure, doodles)[1])


def get_structure_center_points(structure, doodles, samples=20):
    y_min, y_max = structure_y_range(structure, doodles)

    if y_min >= y_max:
        return []

    points = []

    for y in np.linspace(y_min, y_max, samples):
        x = structure_x_at_y(structure, doodles, y)

        if x is not None:
            points.append((y, x))

    return points


def get_two_edge_center_points(left, right, doodles, samples=20):
    y_min = max(structure_y_range(left, doodles)[0], structure_y_range(right, doodles)[0])
    y_max = min(structure_y_range(left, doodles)[1], structure_y_range(right, doodles)[1])

    if y_min >= y_max:
        return []

    points = []

    for y in np.linspace(y_min, y_max, samples):
        left_x = structure_x_at_y(left, doodles, y)
        right_x = structure_x_at_y(right, doodles, y)

        if left_x is not None and right_x is not None:
            points.append((y, (left_x + right_x) / 2))

    return points


def fit_lane_center_curve(center_points):
    if len(center_points) < 2:
        return None

    y_values = np.array([point[0] for point in center_points], dtype=float)
    x_values = np.array([point[1] for point in center_points], dtype=float)

    if len(center_points) == 2:
        return np.polyfit(y_values, x_values, 1)

    linear_coefficients = np.polyfit(y_values, x_values, 1)
    quadratic_coefficients = np.polyfit(y_values, x_values, 2)

    linear_prediction = np.polyval(linear_coefficients, y_values)
    quadratic_prediction = np.polyval(quadratic_coefficients, y_values)

    linear_error = np.mean((x_values - linear_prediction) ** 2)
    quadratic_error = np.mean((x_values - quadratic_prediction) ** 2)

    if quadratic_error < linear_error * 0.5:
        return quadratic_coefficients

    return linear_coefficients


def extend_doodle_to_bottom(doodle, image_width, image_height):
    if len(doodle) < 2:
        return False

    doodle.sort(key=lambda point: point[0])

    side_y, side_x = doodle[-1]

    if side_y >= image_height - 2:
        return False

    neighbour_y, neighbour_x = doodle[-2]
    dy = side_y - neighbour_y
    dx = side_x - neighbour_x

    if dy <= 0:
        return False

    bottom_y = image_height - 1
    t = (bottom_y - side_y) / dy
    bottom_x = side_x + dx * t
    bottom_x = max(0, min(image_width - 1, bottom_x))

    doodle.append((bottom_y, bottom_x))

    return True


def extend_structure_to_bottom(structure, doodles, image_width, image_height):
    extended = False

    for doodle_index in structure:
        if extend_doodle_to_bottom(doodles[doodle_index], image_width, image_height):
            extended = True

    return extended


def extend_outer_edges(structures, doodles, edge_indices, image_width, image_height):
    extended = False

    for index in edge_indices:
        if extend_structure_to_bottom(structures[index], doodles, image_width, image_height):
            extended = True

    return extended


def calculate_hough_lateral_error(lane_center_coefficients, image_width, image_height):
    if lane_center_coefficients is None:
        return None

    lookahead_y = int(image_height * LOOKAHEAD_RATIO)
    lane_center_x = np.polyval(lane_center_coefficients, lookahead_y)
    image_center_x = image_width / 2
    pixel_error = lane_center_x - image_center_x
    normalized_error = pixel_error / image_width

    return normalized_error


def build_color_grid(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    height, width = image.shape[:2]
    rows = int(np.ceil(height / COLOR_GRID_SIZE))
    cols = int(np.ceil(width / COLOR_GRID_SIZE))
    grid = np.zeros((rows, cols, 3), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * COLOR_GRID_SIZE, min((row + 1) * COLOR_GRID_SIZE, height)
            x1, x2 = col * COLOR_GRID_SIZE, min((col + 1) * COLOR_GRID_SIZE, width)
            grid[row, col] = np.mean(lab[y1:y2, x1:x2].reshape(-1, 3), axis=0)

    return grid


def get_road_color_reference(color_grid):
    rows, cols = color_grid.shape[:2]
    half_width = max(1, int(cols * ROAD_REFERENCE_WIDTH / 2))
    start_col = max(0, cols // 2 - half_width)
    end_col = min(cols, cols // 2 + half_width + 1)
    start_row = max(0, int(rows * (1 - ROAD_REFERENCE_HEIGHT)))
    patch = color_grid[start_row:rows, start_col:end_col]

    if patch.size == 0:
        return None

    return np.median(patch.reshape(-1, 3), axis=0)


def find_road_region(color_grid):
    rows, cols = color_grid.shape[:2]
    reference_color = get_road_color_reference(color_grid)

    if reference_color is None:
        return None

    distances = np.linalg.norm(color_grid - reference_color, axis=2)
    road_mask = (distances <= ROAD_COLOR_DISTANCE).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=4)
    valid_labels = []

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        top = stats[label, cv2.CC_STAT_TOP]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        bottom_ratio = (top + height) / rows

        if area >= ROAD_MIN_REGION_CELLS and bottom_ratio >= ROAD_MIN_BOTTOM_RATIO:
            valid_labels.append(label)

    if not valid_labels:
        return None

    # Select only the largest valid connected component.
    best_label = max(valid_labels, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    selected_mask = (labels == best_label)


    road_cells = np.argwhere(selected_mask)

    min_row, min_col = road_cells.min(axis=0)
    max_row, max_col = road_cells.max(axis=0)

    return {
        "mask": selected_mask.astype(np.uint8),
        "cells": [tuple(cell) for cell in road_cells],
        "area": len(road_cells),
        "min_row": int(min_row),
        "max_row": int(max_row),
        "min_col": int(min_col),
        "max_col": int(max_col),
        "reference_color": reference_color
    }


def detect_road_region(image):
    color_grid = build_color_grid(image)
    road_region = find_road_region(color_grid)
    return color_grid, road_region


def track_ground_corridor(region):
    if region is None:
        return None

    mask = region["mask"]
    rows, cols = mask.shape
    intervals = {}

    # Find contiguous ground intervals in every grid row.
    for row in range(rows):
        ground_cols = np.flatnonzero(mask[row])
        if len(ground_cols) == 0:
            continue

        splits = np.where(np.diff(ground_cols) > 1)[0] + 1
        groups = np.split(ground_cols, splits)
        intervals[row] = [(int(group[0]), int(group[-1])) for group in groups]

    if not intervals:
        return None

    # Find a plausible corridor near the bottom, ignoring full-width regions.
    image_center = (cols - 1) / 2
    max_start_width = cols * 0.85
    start_row = None
    selected = None

    for row in sorted(intervals.keys(), reverse=True):
        candidates = [
            item for item in intervals[row]
            if item[1] - item[0] + 1 < max_start_width
        ]
        if not candidates:
            continue

        selected = min(
            candidates,
            key=lambda item: abs((item[0] + item[1]) / 2 - image_center)
        )
        start_row = row
        break

    if selected is None:
        return {}, intervals

    left, right = selected
    previous_center = (left + right) / 2
    previous_width = right - left + 1
    corridor = {start_row: selected}

    # Track upward, tolerating small gaps and perspective-related width changes.
    max_center_shift = max(3, cols * 0.06)
    max_width_change = max(5, previous_width * 0.30)
    max_missing_rows = 2
    missing_rows = 0

    for row in range(start_row - 1, min(intervals) - 1, -1):
        candidates = intervals.get(row, [])

        if not candidates:
            missing_rows += 1
            if missing_rows > max_missing_rows:
                break
            continue

        valid = []
        for candidate_left, candidate_right in candidates:
            center = (candidate_left + candidate_right) / 2
            width = candidate_right - candidate_left + 1
            center_shift = abs(center - previous_center)
            width_change = abs(width - previous_width)

            if center_shift <= max_center_shift and width_change <= max_width_change:
                score = center_shift + width_change * 0.5
                valid.append((score, candidate_left, candidate_right))

        if not valid:
            missing_rows += 1
            if missing_rows > max_missing_rows:
                break
            continue

        _, left, right = min(valid)
        missing_rows = 0
        previous_center = (left + right) / 2
        previous_width = right - left + 1
        corridor[row] = (left, right)

    return corridor, intervals


def calculate_color_lateral_error(region, corridor, image_width, image_height):
    if region is None or not corridor:
        return None

    lookahead_y = int(image_height * LOOKAHEAD_RATIO)
    target_row = lookahead_y / COLOR_GRID_SIZE - 0.5
    tracked_rows = sorted(corridor)

    # Do not extrapolate beyond the detected corridor.
    if target_row < tracked_rows[0] or target_row > tracked_rows[-1]:
        return None

    lower_rows = [row for row in tracked_rows if row <= target_row]
    upper_rows = [row for row in tracked_rows if row >= target_row]
    lower = max(lower_rows)
    upper = min(upper_rows)

    left_lower, right_lower = corridor[lower]
    left_upper, right_upper = corridor[upper]

    if lower == upper:
        left, right = left_lower, right_lower
    else:
        ratio = (target_row - lower) / (upper - lower)
        left = left_lower + ratio * (left_upper - left_lower)
        right = right_lower + ratio * (right_upper - right_lower)

    lane_center_x = ((left + right + 1) / 2) * COLOR_GRID_SIZE
    lane_center_x = min(max(lane_center_x, 0), image_width - 1)
    pixel_error = lane_center_x - image_width / 2
    normalized_error = pixel_error / image_width

    return normalized_error, lookahead_y, lane_center_x, pixel_error


def detect_lane(image):
    image = resize_image(image)
    height, width = image.shape[:2]

    roi = create_roi(image)
    edges = preprocess(roi)
    lines = get_hough_lines(edges)
    groups = group_continuous_lines(lines)
    doodles = build_doodles(groups)
    structures = build_structure_groups(doodles)

    structures = sort_structures_left_to_right(structures, doodles)

    if len(structures) >= 2:
        extend_outer_edges(structures, doodles, [0, len(structures) - 1], width, height)

    structures = filter_weak_structures(structures, doodles)

    lane_center_coefficients = None
    center_points = None
    detection_type = None

    if len(structures) == 3:
        structures = sort_structures_left_to_right(structures, doodles)

        if preliminary_candidate_is_valid(structures, doodles):
            extend_outer_edges(structures, doodles, [0, 2], width, height)
            center_points = get_structure_center_points(structures[1], doodles)
            lane_center_coefficients = fit_lane_center_curve(center_points)
            detection_type = "three_structure"

    elif len(structures) == 2:
        structures = sorted(structures, key=lambda structure: structure_x_range(structure, doodles)[1])

        extend_outer_edges(structures, doodles, [0, 1], width, height)
        center_points = get_two_edge_center_points(structures[0], structures[1], doodles)
        lane_center_coefficients = fit_lane_center_curve(center_points)
        detection_type = "two_edge"

    else:
        structures = merge_structures(structures, doodles)
        candidates = generate_structure_candidates(structures, doodles)
        candidates = filter_candidates_by_shared_height(candidates, structures, doodles)

        best_candidate = None
        best_error = float("inf")

        for candidate in candidates:
            left, middle, right = candidate
            error = evaluate_candidate(
                structures[left],
                structures[middle],
                structures[right],
                doodles
            )

            if error is not None and error < best_error:
                best_error = error
                best_candidate = candidate

        if best_candidate is not None:
            left_index, middle_index, right_index = best_candidate
            edge_indices = [left_index, right_index]

            edge_indices = [
                index for index in edge_indices
                if not is_image_border_structure(structures[index], doodles, width)
            ]

            extend_outer_edges(structures, doodles, edge_indices, width, height)
            center_points = get_structure_center_points(structures[middle_index], doodles)
            lane_center_coefficients = fit_lane_center_curve(center_points)
            detection_type = "three_structure_candidate"

    # hough_offset = calculate_hough_lateral_error(lane_center_coefficients, width, height)

    _, road_region = detect_road_region(image)
    tracked = track_ground_corridor(road_region)

    corridor = tracked[0] if tracked is not None else None
    color_result = calculate_color_lateral_error(road_region, corridor, width, height)

    normalized_error = color_result[0] if color_result is not None else None
    color_detected = normalized_error is not None

    return {
        "lane_offset": normalized_error,
        "detected": color_detected,
        "detection_type": "color_corridor" if color_detected else None,
        "image_width": width,
        "image_height": height,
        # "hough_lane_offset": hough_offset,
        # "hough_detection_type": detection_type
    }